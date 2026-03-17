#include <chrono>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/string.hpp"

#include <leptonica/allheaders.h>
#include <tesseract/baseapi.h>

using namespace std::chrono_literals;

class OcrReaderCppNode final : public rclcpp::Node
{
public:
  OcrReaderCppNode() : Node("ocr_reader")
  {
    image_path_ = this->declare_parameter<std::string>("image_path", "/home/lu/code/ocr/input.png");
    lang_ = this->declare_parameter<std::string>("lang", "eng");
    psm_ = this->declare_parameter<int>("psm", 6);  // 6=single uniform block
    invert_ = this->declare_parameter<bool>("invert", false);
    scale_ = this->declare_parameter<double>("scale", 2.0);
    whitelist_ = this->declare_parameter<std::string>(
      "whitelist",
      "0123456789+-*/().=xX\xC3\x97\xC3\xB7");  // includes × and ÷ (UTF-8)
    blacklist_ = this->declare_parameter<std::string>("blacklist", "");
    period_ms_ = this->declare_parameter<int>("period_ms", 500);
    topic_ = this->declare_parameter<std::string>("topic", "ocr_text");

    pub_ = this->create_publisher<std_msgs::msg::String>(topic_, 10);

    if (tess_.Init(nullptr, lang_.c_str()) != 0) {
      throw std::runtime_error("tesseract Init failed (lang=" + lang_ + ")");
    }

    (void)tess_.SetVariable("tessedit_char_whitelist", whitelist_.c_str());
    if (!blacklist_.empty()) {
      (void)tess_.SetVariable("tessedit_char_blacklist", blacklist_.c_str());
    }
    tess_.SetPageSegMode(static_cast<tesseract::PageSegMode>(psm_));

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(period_ms_),
      std::bind(&OcrReaderCppNode::tick, this));
  }

  ~OcrReaderCppNode() override
  {
    tess_.End();
  }

private:
  void tick()
  {
    Pix * raw = pixRead(image_path_.c_str());
    if (raw == nullptr) {
      RCLCPP_WARN(get_logger(), "Failed to read image: %s", image_path_.c_str());
      return;
    }

    // Pix * img8 = pixConvertTo8(raw, 0);
    // pixDestroy(&raw);
    // if (img8 == nullptr) {
    //   RCLCPP_WARN(get_logger(), "Failed to convert image to 8bpp: %s", image_path_.c_str());
    //   return;
    // }

    // Pix * proc = img8;
    // if (invert_) {
    //   Pix * inv = pixInvert(nullptr, proc);
    //   pixDestroy(&proc);
    //   proc = inv;
    // }

    // if (scale_ > 1.01) {
    //   Pix * scaled = pixScale(proc, scale_, scale_);
    //   pixDestroy(&proc);
    //   proc = scaled;
    // }

    // if (proc == nullptr) {
    //   RCLCPP_WARN(get_logger(), "Preprocess failed for image: %s", image_path_.c_str());
    //   return;
    // }

    tess_.SetImage(raw);

    char * out = tess_.GetUTF8Text();
    std::string text = (out != nullptr) ? std::string(out) : std::string();

    if (out != nullptr) {
      delete[] out;
    }

    auto msg = std_msgs::msg::String();
    msg.data = text;
    pub_->publish(msg);

    std::string preview = text;
    for (char & c : preview) {
      if (c == '\n') c = ' ';
    }
    if (preview.size() > 200) {
      preview = preview.substr(0, 200) + "...";
    }

    RCLCPP_INFO(get_logger(), "OCR published (%zu chars): %s", text.size(), preview.c_str());

    tess_.Clear();
    // pixDestroy(&proc);
  }

  std::string image_path_;
  std::string lang_;
  int psm_ = 6;
  bool invert_ = true;
  double scale_ = 2.0;
  std::string whitelist_;
  std::string blacklist_;
  int period_ms_ = 500;
  std::string topic_;

  tesseract::TessBaseAPI tess_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<OcrReaderCppNode>());
  rclcpp::shutdown();
  return 0;
}
