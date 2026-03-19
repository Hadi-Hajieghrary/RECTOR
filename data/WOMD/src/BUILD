# BUILD file for RECTOR data conversion tools
# Builds conversion tool linking against Waymo Open Dataset libraries

cc_binary(
    name = "convert_scenario_to_tf_example",
    srcs = ["convert_scenario_to_tf_example.cc"],
    deps = [
        "@waymo_open_dataset//waymo_open_dataset/data_conversion:scenario_conversion",
        "@waymo_open_dataset//waymo_open_dataset/protos:scenario_proto",
        "@waymo_open_dataset//waymo_open_dataset/protos:conversion_config_proto",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "@com_google_absl//absl/status:statusor",
        "@org_tensorflow//tensorflow/core:lib",
        "@org_tensorflow//tensorflow/core:protos_all_cc",
    ],
)
