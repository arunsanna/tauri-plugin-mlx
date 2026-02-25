// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "MLXInference",
    platforms: [
        .macOS(.v14)  // MLX requires macOS 14+; works on macOS 26 (Tahoe)
    ],
    products: [
        .library(
            name: "MLXInference",
            type: .static,
            targets: ["MLXInference"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", branch: "main"),
    ],
    targets: [
        .target(
            name: "MLXInference",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
            ],
            path: "Sources"
        )
    ]
)
