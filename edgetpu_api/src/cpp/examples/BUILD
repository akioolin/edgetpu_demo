cc_library(
    name = "model_utils",
    srcs = ["model_utils.cc"],
    hdrs = ["model_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:builtin_op_data",
        "@org_tensorflow//tensorflow/lite:framework",
        "@org_tensorflow//tensorflow/lite/kernels:builtin_ops",
    ],
)

cc_library(
    name = "label_utils",
    srcs = ["label_utils.cc"],
    hdrs = ["label_utils.h"],
    deps = [
        "@com_google_absl//absl/strings",
    ],
)

cc_binary(
    name = "two_models_one_tpu",
    testonly = 1,
    srcs = ["two_models_one_tpu.cc"],
    deps = [
        ":model_utils",
        "//src/cpp:test_utils",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "two_models_two_tpus_threaded",
    testonly = 1,
    srcs = ["two_models_two_tpus_threaded.cc"],
    deps = [
        ":model_utils",
        "//src/cpp:test_utils",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "minimal",
    testonly = 1,
    srcs = ["minimal.cc"],
    deps = [
        ":model_utils",
        "@libedgetpu//:header",
        "@org_tensorflow//tensorflow/lite:framework",
    ],
)

cc_binary(
    name = "classify_image",
    testonly = 1,
    srcs = ["classify_image.cc"],
    deps = [
        ":label_utils",
        ":model_utils",
        "//src/cpp:test_utils",
        "//src/cpp/classification:engine",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
    ],
)
