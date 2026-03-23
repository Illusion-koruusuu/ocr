#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"
#include "onnxruntime_cxx_api.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std::chrono_literals;

class TjoabOnnxSmokeNode final : public rclcpp::Node
{
public:
  TjoabOnnxSmokeNode()
  : Node("tjoab_onnx_smoke_node"),
    env_(ORT_LOGGING_LEVEL_WARNING, "tjoab_onnx_smoke")
  {
    model_dir_ = this->declare_parameter<std::string>(
      "model_dir", "/home/lu/code/ocr/onnx/tjoab_latex_finetuned");
    image_path_ = this->declare_parameter<std::string>(
      "image_path", "/home/lu/code/ocr/png/11/roi_36.png");
    invert_image_ = this->declare_parameter<bool>("invert_image", true);
    max_length_ = this->declare_parameter<int>("max_length", 128);
    decoder_start_token_id_ = this->declare_parameter<int>("decoder_start_token_id", 2);
    eos_token_id_ = this->declare_parameter<int>("eos_token_id", 2);
    topic_ = this->declare_parameter<std::string>("topic", "tjoab_token_ids");
    text_topic_ = this->declare_parameter<std::string>("text_topic", "tjoab_text");
    id2token_path_ = this->declare_parameter<std::string>(
      "id2token_path", "/home/lu/code/ocr/onnx/tjoab_latex_finetuned/id2token.txt");
    period_ms_ = this->declare_parameter<int>("period_ms", 3000);

    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options_.SetIntraOpNumThreads(0);

    const std::string encoder_path = model_dir_ + "/encoder_model.onnx";
    const std::string decoder_path = model_dir_ + "/decoder_model.onnx";
    const std::string decoder_with_past_path = model_dir_ + "/decoder_with_past_model.onnx";

    if (!std::filesystem::exists(encoder_path) ||
      !std::filesystem::exists(decoder_path) ||
      !std::filesystem::exists(decoder_with_past_path))
    {
      throw std::runtime_error("ONNX files not found under model_dir=" + model_dir_);
    }

    encoder_session_ = std::make_unique<Ort::Session>(env_, encoder_path.c_str(), session_options_);
    decoder_session_ = std::make_unique<Ort::Session>(env_, decoder_path.c_str(), session_options_);
    decoder_with_past_session_ = std::make_unique<Ort::Session>(
      env_, decoder_with_past_path.c_str(), session_options_);

    RCLCPP_INFO(this->get_logger(), "Loaded ONNX sessions from %s", model_dir_.c_str());
    log_session_io(*encoder_session_, "encoder");
    log_session_io(*decoder_session_, "decoder");
    log_session_io(*decoder_with_past_session_, "decoder_with_past");

    publisher_ = this->create_publisher<std_msgs::msg::String>(topic_, 10);
    text_publisher_ = this->create_publisher<std_msgs::msg::String>(text_topic_, 10);

    load_id2token_map(id2token_path_);
    init_byte_level_decoder();

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(period_ms_),
      std::bind(&TjoabOnnxSmokeNode::tick, this));
  }

private:
  struct CacheTensor
  {
    std::vector<float> data;
    std::vector<int64_t> shape;
  };

  static std::vector<uint32_t> utf8_to_codepoints(const std::string & s)
  {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
      unsigned char c = static_cast<unsigned char>(s[i]);
      if (c < 0x80) {
        out.push_back(c);
        ++i;
      } else if ((c >> 5) == 0x6 && i + 1 < s.size()) {
        uint32_t cp = ((c & 0x1F) << 6) | (static_cast<unsigned char>(s[i + 1]) & 0x3F);
        out.push_back(cp);
        i += 2;
      } else if ((c >> 4) == 0xE && i + 2 < s.size()) {
        uint32_t cp = ((c & 0x0F) << 12) |
          ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 6) |
          (static_cast<unsigned char>(s[i + 2]) & 0x3F);
        out.push_back(cp);
        i += 3;
      } else if ((c >> 3) == 0x1E && i + 3 < s.size()) {
        uint32_t cp = ((c & 0x07) << 18) |
          ((static_cast<unsigned char>(s[i + 1]) & 0x3F) << 12) |
          ((static_cast<unsigned char>(s[i + 2]) & 0x3F) << 6) |
          (static_cast<unsigned char>(s[i + 3]) & 0x3F);
        out.push_back(cp);
        i += 4;
      } else {
        out.push_back('?');
        ++i;
      }
    }
    return out;
  }

  static std::string codepoint_to_utf8(uint32_t cp)
  {
    std::string out;
    if (cp <= 0x7F) {
      out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
      out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
      out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
      out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
      out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
  }

  void init_byte_level_decoder()
  {
    std::set<int> bs;
    for (int i = 33; i <= 126; ++i) {
      bs.insert(i);
    }
    for (int i = 161; i <= 172; ++i) {
      bs.insert(i);
    }
    for (int i = 174; i <= 255; ++i) {
      bs.insert(i);
    }

    std::vector<int> bs_vec(bs.begin(), bs.end());
    std::vector<int> cs = bs_vec;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
      if (bs.find(b) == bs.end()) {
        bs_vec.push_back(b);
        cs.push_back(256 + n);
        ++n;
      }
    }

    byte_decoder_.clear();
    for (size_t i = 0; i < bs_vec.size(); ++i) {
      byte_decoder_[static_cast<uint32_t>(cs[i])] = static_cast<uint8_t>(bs_vec[i]);
    }
  }

  void load_id2token_map(const std::string & path)
  {
    std::ifstream in(path);
    if (!in.is_open()) {
      RCLCPP_WARN(this->get_logger(), "id2token file not found: %s", path.c_str());
      return;
    }

    std::vector<std::string> local;
    std::string line;
    while (std::getline(in, line)) {
      auto pos = line.find('\t');
      if (pos == std::string::npos) {
        continue;
      }
      int id = std::stoi(line.substr(0, pos));
      std::string token = line.substr(pos + 1);
      if (id >= static_cast<int>(local.size())) {
        local.resize(static_cast<size_t>(id + 1));
      }
      local[static_cast<size_t>(id)] = token;
    }

    id2token_ = std::move(local);
    RCLCPP_INFO(this->get_logger(), "Loaded id2token entries: %zu", id2token_.size());
  }

  std::string decode_token_ids(const std::vector<int64_t> & token_ids)
  {
    if (id2token_.empty()) {
      return "";
    }

    std::string joined;
    for (int64_t id : token_ids) {
      if (id == 0 || id == 1 || id == 2) {
        continue;
      }
      if (id < 0 || static_cast<size_t>(id) >= id2token_.size()) {
        continue;
      }
      joined += id2token_[static_cast<size_t>(id)];
    }

    // ByteLevel decoder: token string -> byte sequence -> UTF-8 text
    std::vector<uint8_t> bytes;
    for (uint32_t cp : utf8_to_codepoints(joined)) {
      auto it = byte_decoder_.find(cp);
      if (it != byte_decoder_.end()) {
        bytes.push_back(it->second);
      } else {
        std::string utf8 = codepoint_to_utf8(cp);
        bytes.insert(bytes.end(), utf8.begin(), utf8.end());
      }
    }

    return std::string(bytes.begin(), bytes.end());
  }

  static CacheTensor copy_tensor_to_cache(Ort::Value & value)
  {
    CacheTensor out;
    auto info = value.GetTensorTypeAndShapeInfo();
    out.shape = info.GetShape();
    size_t count = info.GetElementCount();
    const float * ptr = value.GetTensorData<float>();
    out.data.assign(ptr, ptr + count);
    return out;
  }

  static std::vector<std::string> get_input_names(Ort::Session & session)
  {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = session.GetInputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      auto name = session.GetInputNameAllocated(i, allocator);
      names.emplace_back(name.get());
    }
    return names;
  }

  static std::vector<std::string> get_output_names(Ort::Session & session)
  {
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t count = session.GetOutputCount();
    std::vector<std::string> names;
    names.reserve(count);
    for (size_t i = 0; i < count; ++i) {
      auto name = session.GetOutputNameAllocated(i, allocator);
      names.emplace_back(name.get());
    }
    return names;
  }

  void log_session_io(Ort::Session & session, const std::string & tag)
  {
    const auto input_names = get_input_names(session);
    const auto output_names = get_output_names(session);

    std::string in;
    for (size_t i = 0; i < input_names.size(); ++i) {
      in += input_names[i];
      if (i + 1 < input_names.size()) {
        in += ", ";
      }
    }

    std::string out;
    for (size_t i = 0; i < output_names.size(); ++i) {
      out += output_names[i];
      if (i + 1 < output_names.size()) {
        out += ", ";
      }
    }

    RCLCPP_INFO(this->get_logger(), "[%s] inputs: %s", tag.c_str(), in.c_str());
    RCLCPP_INFO(this->get_logger(), "[%s] outputs: %s", tag.c_str(), out.c_str());
  }

  void tick()
  {
    try {
      const auto t0 = std::chrono::steady_clock::now();

      // 1) encoder forward with a real preprocessed image tensor [1, 3, 384, 384]
      std::vector<float> pixel_values = load_pixel_values(image_path_, invert_image_);
      const std::array<int64_t, 4> pixel_shape = {1, 3, 384, 384};

      Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value pixel_tensor = Ort::Value::CreateTensor<float>(
        mem,
        pixel_values.data(),
        pixel_values.size(),
        pixel_shape.data(),
        pixel_shape.size());

      const auto encoder_inputs = get_input_names(*encoder_session_);
      const auto encoder_outputs = get_output_names(*encoder_session_);
      std::vector<const char *> encoder_input_names{encoder_inputs[0].c_str()};
      std::vector<const char *> encoder_output_names{encoder_outputs[0].c_str()};
      std::vector<Ort::Value> encoder_input_tensors;
      encoder_input_tensors.emplace_back(std::move(pixel_tensor));

      auto encoder_run_outputs = encoder_session_->Run(
        Ort::RunOptions{nullptr},
        encoder_input_names.data(),
        encoder_input_tensors.data(),
        encoder_input_tensors.size(),
        encoder_output_names.data(),
        encoder_output_names.size());

      auto & encoder_hidden = encoder_run_outputs[0];
      const auto enc_shape = encoder_hidden.GetTensorTypeAndShapeInfo().GetShape();
      if (enc_shape.size() != 3 || enc_shape[0] != 1) {
        throw std::runtime_error("Unexpected encoder output shape");
      }

      encoder_hidden_shape_ = {enc_shape[0], enc_shape[1], enc_shape[2]};
      const size_t enc_count = static_cast<size_t>(enc_shape[0] * enc_shape[1] * enc_shape[2]);
      const float * enc_data = encoder_hidden.GetTensorData<float>();
      encoder_hidden_data_.assign(enc_data, enc_data + enc_count);

      const auto decoder_inputs = get_input_names(*decoder_session_);
      const auto decoder_outputs = get_output_names(*decoder_session_);
      const auto decoder_with_past_inputs = get_input_names(*decoder_with_past_session_);
      const auto decoder_with_past_outputs = get_output_names(*decoder_with_past_session_);

      // 2) greedy decode with KV cache:
      // first step uses decoder_model, following steps use decoder_with_past.
      std::vector<int64_t> token_ids;
      token_ids.reserve(static_cast<size_t>(max_length_));
      token_ids.push_back(static_cast<int64_t>(decoder_start_token_id_));

      std::vector<const char *> decoder_input_names{
        decoder_inputs[0].c_str(),
        decoder_inputs[1].c_str()};
      std::vector<const char *> decoder_output_names;
      decoder_output_names.reserve(decoder_outputs.size());
      for (const auto & n : decoder_outputs) {
        decoder_output_names.push_back(n.c_str());
      }

      // First decode step with decoder_model.onnx
      const std::array<int64_t, 2> first_ids_shape = {1, 1};
      Ort::Value first_ids_tensor = Ort::Value::CreateTensor<int64_t>(
        mem,
        token_ids.data(),
        token_ids.size(),
        first_ids_shape.data(),
        first_ids_shape.size());

      std::vector<Ort::Value> first_decoder_inputs;
      first_decoder_inputs.emplace_back(std::move(first_ids_tensor));
      first_decoder_inputs.emplace_back(Ort::Value::CreateTensor<float>(
        mem,
        encoder_hidden_data_.data(),
        encoder_hidden_data_.size(),
        encoder_hidden_shape_.data(),
        encoder_hidden_shape_.size()));

      auto first_decoder_outputs = decoder_session_->Run(
        Ort::RunOptions{nullptr},
        decoder_input_names.data(),
        first_decoder_inputs.data(),
        first_decoder_inputs.size(),
        decoder_output_names.data(),
        decoder_output_names.size());

      auto & first_logits = first_decoder_outputs[0];
      auto first_shape = first_logits.GetTensorTypeAndShapeInfo().GetShape();
      if (first_shape.size() != 3 || first_shape[0] != 1 || first_shape[1] != 1) {
        throw std::runtime_error("Unexpected first-step logits shape from decoder_model.onnx");
      }

      {
        const int64_t vocab_size = first_shape[2];
        const float * data = first_logits.GetTensorData<float>();
        auto max_it = std::max_element(data, data + vocab_size);
        const int64_t next_token_id = static_cast<int64_t>(std::distance(data, max_it));
        token_ids.push_back(next_token_id);
      }

      if (token_ids.back() != eos_token_id_ && max_length_ > 1) {
        // decoder_with_past expects 48 KV tensors per step:
        // 12 layers * (decoder.key, decoder.value, encoder.key, encoder.value).
        // decoder KV is updated each step; encoder KV stays constant from first step.
        constexpr int kNumLayers = 12;
        std::vector<CacheTensor> decoder_kv_cache;
        std::vector<CacheTensor> encoder_kv_cache;
        decoder_kv_cache.reserve(kNumLayers * 2);
        encoder_kv_cache.reserve(kNumLayers * 2);

        for (int layer = 0; layer < kNumLayers; ++layer) {
          const size_t base = 1 + static_cast<size_t>(layer) * 4;
          decoder_kv_cache.emplace_back(copy_tensor_to_cache(first_decoder_outputs[base + 0]));
          decoder_kv_cache.emplace_back(copy_tensor_to_cache(first_decoder_outputs[base + 1]));
          encoder_kv_cache.emplace_back(copy_tensor_to_cache(first_decoder_outputs[base + 2]));
          encoder_kv_cache.emplace_back(copy_tensor_to_cache(first_decoder_outputs[base + 3]));
        }

        std::vector<const char *> dwp_input_names;
        dwp_input_names.reserve(decoder_with_past_inputs.size());
        for (const auto & n : decoder_with_past_inputs) {
          dwp_input_names.push_back(n.c_str());
        }

        std::vector<const char *> dwp_output_names;
        dwp_output_names.reserve(decoder_with_past_outputs.size());
        for (const auto & n : decoder_with_past_outputs) {
          dwp_output_names.push_back(n.c_str());
        }

        for (int step = 1; step < max_length_ - 1; ++step) {
          int64_t last_token = token_ids.back();
          const std::array<int64_t, 2> ids_shape = {1, 1};
          Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
            mem,
            &last_token,
            1,
            ids_shape.data(),
            ids_shape.size());

          std::vector<Ort::Value> dwp_inputs;
          dwp_inputs.reserve(1 + static_cast<size_t>(kNumLayers) * 4);
          dwp_inputs.emplace_back(std::move(ids_tensor));

          for (int layer = 0; layer < kNumLayers; ++layer) {
            const size_t d = static_cast<size_t>(layer) * 2;
            dwp_inputs.emplace_back(Ort::Value::CreateTensor<float>(
              mem,
              decoder_kv_cache[d + 0].data.data(),
              decoder_kv_cache[d + 0].data.size(),
              decoder_kv_cache[d + 0].shape.data(),
              decoder_kv_cache[d + 0].shape.size()));
            dwp_inputs.emplace_back(Ort::Value::CreateTensor<float>(
              mem,
              decoder_kv_cache[d + 1].data.data(),
              decoder_kv_cache[d + 1].data.size(),
              decoder_kv_cache[d + 1].shape.data(),
              decoder_kv_cache[d + 1].shape.size()));
            dwp_inputs.emplace_back(Ort::Value::CreateTensor<float>(
              mem,
              encoder_kv_cache[d + 0].data.data(),
              encoder_kv_cache[d + 0].data.size(),
              encoder_kv_cache[d + 0].shape.data(),
              encoder_kv_cache[d + 0].shape.size()));
            dwp_inputs.emplace_back(Ort::Value::CreateTensor<float>(
              mem,
              encoder_kv_cache[d + 1].data.data(),
              encoder_kv_cache[d + 1].data.size(),
              encoder_kv_cache[d + 1].shape.data(),
              encoder_kv_cache[d + 1].shape.size()));
          }

          auto dwp_outputs = decoder_with_past_session_->Run(
            Ort::RunOptions{nullptr},
            dwp_input_names.data(),
            dwp_inputs.data(),
            dwp_inputs.size(),
            dwp_output_names.data(),
            dwp_output_names.size());

          auto & logits = dwp_outputs[0];
          auto shape = logits.GetTensorTypeAndShapeInfo().GetShape();
          if (shape.size() != 3 || shape[0] != 1 || shape[1] != 1) {
            throw std::runtime_error("Unexpected logits shape from decoder_with_past_model.onnx");
          }

          const int64_t vocab_size = shape[2];
          const float * data = logits.GetTensorData<float>();
          auto max_it = std::max_element(data, data + vocab_size);
          const int64_t next_token_id = static_cast<int64_t>(std::distance(data, max_it));
          token_ids.push_back(next_token_id);

          // decoder_with_past outputs only decoder KV (2 tensors per layer), update those.
          for (int layer = 0; layer < kNumLayers; ++layer) {
            const size_t d = static_cast<size_t>(layer) * 2;
            const size_t out_base = 1 + d;
            decoder_kv_cache[d + 0] = copy_tensor_to_cache(dwp_outputs[out_base + 0]);
            decoder_kv_cache[d + 1] = copy_tensor_to_cache(dwp_outputs[out_base + 1]);
          }

          if (next_token_id == eos_token_id_) {
            break;
          }
        }
      }

      std::ostringstream oss;
      for (size_t i = 0; i < token_ids.size(); ++i) {
        if (i > 0) {
          oss << " ";
        }
        oss << token_ids[i];
      }

      std_msgs::msg::String msg;
      msg.data = oss.str();
      publisher_->publish(msg);

      const std::string decoded_text = decode_token_ids(token_ids);
      if (!decoded_text.empty()) {
        std_msgs::msg::String text_msg;
        text_msg.data = decoded_text;
        text_publisher_->publish(text_msg);
      }

      const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t0).count();

      RCLCPP_INFO(
        this->get_logger(),
        "ONNX decode ok: image=%s, tokens=%zu, text_len=%zu, elapsed=%ld ms",
        image_path_.c_str(),
        token_ids.size(),
        decoded_text.size(),
        elapsed_ms);
    } catch (const std::exception & e) {
      RCLCPP_ERROR(this->get_logger(), "ONNX smoke run failed: %s", e.what());
    }
  }

  static std::vector<float> load_pixel_values(const std::string & image_path, bool invert_image)
  {
    cv::Mat bgr = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bgr.empty()) {
      throw std::runtime_error("Failed to read image: " + image_path);
    }

    if (invert_image) {
      cv::bitwise_not(bgr, bgr);
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(384, 384), 0.0, 0.0, cv::INTER_LINEAR);

    cv::Mat f32;
    resized.convertTo(f32, CV_32FC3, 2.0 / 255.0, -1.0);

    std::vector<float> pixel_values(1 * 3 * 384 * 384);
    size_t idx = 0;
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 384; ++h) {
        for (int w = 0; w < 384; ++w) {
          pixel_values[idx++] = f32.at<cv::Vec3f>(h, w)[c];
        }
      }
    }
    return pixel_values;
  }

  std::string model_dir_;
  std::string image_path_;
  bool invert_image_ = true;
  int max_length_ = 128;
  int decoder_start_token_id_ = 2;
  int eos_token_id_ = 2;
  std::string topic_;
  std::string text_topic_;
  std::string id2token_path_;
  int period_ms_ = 3000;

  // Scratch buffers to avoid dangling encoder output memory in decode loop.
  std::vector<float> encoder_hidden_data_;
  std::array<int64_t, 3> encoder_hidden_shape_{};

  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Session> encoder_session_;
  std::unique_ptr<Ort::Session> decoder_session_;
  std::unique_ptr<Ort::Session> decoder_with_past_session_;

  std::vector<std::string> id2token_;
  std::unordered_map<uint32_t, uint8_t> byte_decoder_;

  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr text_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TjoabOnnxSmokeNode>());
  rclcpp::shutdown();
  return 0;
}
