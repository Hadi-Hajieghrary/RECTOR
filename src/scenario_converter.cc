/* Copyright 2025 RECTOR Project
 *
 * Command-line tool to convert Waymo scenario format to TF Example format
 * This is a standalone wrapper around Waymo's ScenarioToExample function
 */

#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/statusor.h"
#include "tensorflow/core/example/example.pb.h"
#include "tensorflow/core/lib/io/record_writer.h"
#include "tensorflow/core/lib/io/record_reader.h"
#include "waymo_open_dataset/data_conversion/scenario_conversion.h"
#include "waymo_open_dataset/protos/scenario.pb.h"
#include "waymo_open_dataset/protos/conversion_config.pb.h"

ABSL_FLAG(std::string, input, "", "Input scenario TFRecord file path");
ABSL_FLAG(std::string, output, "", "Output TF (TensorFlow Example) TFRecord file path");

namespace rector {
namespace data_conversion {

absl::Status ConvertScenarioFile(const std::string& input_path,
                                 const std::string& output_path,
                                 const waymo::open_dataset::MotionExampleConversionConfig& config) {
  // Open input file
  std::unique_ptr<tensorflow::RandomAccessFile> input_file;
  auto status = tensorflow::Env::Default()->NewRandomAccessFile(input_path, &input_file);
  if (!status.ok()) {
    return absl::InternalError("Failed to open input file: " + input_path);
  }
  tensorflow::io::RecordReader reader(input_file.get());

  // Open output file
  std::unique_ptr<tensorflow::WritableFile> output_file;
  status = tensorflow::Env::Default()->NewWritableFile(output_path, &output_file);
  if (!status.ok()) {
    return absl::InternalError("Failed to open output file: " + output_path);
  }
  tensorflow::io::RecordWriter writer(output_file.get());

  uint64_t offset = 0;
  tensorflow::tstring record;
  int scenario_count = 0;
  int success_count = 0;

  std::map<std::string, int> counters;

  // Read each scenario and convert to Example using Waymo's function
  while (reader.ReadRecord(&offset, &record).ok()) {
    waymo::open_dataset::Scenario scenario;
    if (!scenario.ParseFromString(record)) {
      std::cerr << "Failed to parse scenario " << scenario_count << std::endl;
      continue;
    }

    scenario_count++;

    // Convert scenario to Example using Waymo's official function
    auto result = waymo::open_dataset::ScenarioToExample(scenario, config, &counters);
    if (!result.ok()) {
      std::cerr << "Failed to convert scenario " << scenario.scenario_id()
                << ": " << result.status() << std::endl;
      continue;
    }

    // Write Example to output
    std::string example_str;
    if (!result.value().SerializeToString(&example_str)) {
      std::cerr << "Failed to serialize example for scenario "
                << scenario.scenario_id() << std::endl;
      continue;
    }

    auto write_status = writer.WriteRecord(example_str);
    if (!write_status.ok()) {
      std::cerr << "Failed to write example for scenario "
                << scenario.scenario_id() << std::endl;
      continue;
    }
    success_count++;

    if (scenario_count % 100 == 0) {
      std::cout << "Processed " << scenario_count << " scenarios, "
                << success_count << " successful" << std::endl;
    }
  }

  std::cout << "Conversion complete: " << scenario_count << " scenarios, "
            << success_count << " converted successfully" << std::endl;

  // Print counters
  std::cout << "\nConversion statistics:" << std::endl;
  for (const auto& [key, value] : counters) {
    std::cout << "  " << key << ": " << value << std::endl;
  }

  auto close_status = writer.Close();
  if (!close_status.ok()) {
    return absl::InternalError("Failed to close output file");
  }

  return absl::OkStatus();
}

}  // namespace data_conversion
}  // namespace rector

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  std::string input = absl::GetFlag(FLAGS_input);
  std::string output = absl::GetFlag(FLAGS_output);

  if (input.empty() || output.empty()) {
    std::cerr << "Usage: " << argv[0]
              << " --input=<scenario.tfrecord> --output=<example.tfrecord>" << std::endl;
    return 1;
  }

  // Setup conversion config with defaults
  waymo::open_dataset::MotionExampleConversionConfig config;
  // Using defaults: max_num_agents=128, num_past_steps=11, num_future_steps=80

  auto status = rector::data_conversion::ConvertScenarioFile(input, output, config);
  if (!status.ok()) {
    std::cerr << "Conversion failed: " << status << std::endl;
    return 1;
  }

  return 0;
}
