# Distribution and external dependencies

The project owner must choose a license for project-owned code before an open-source release. No license grant for third-party software, robot models, or course materials is implied by this deployment package.

- NVIDIA Isaac Gym Preview 4 must be obtained by each operator from [NVIDIA's archive](https://developer.nvidia.com/isaac-gym/download) under its own license. It is not bundled or downloaded automatically. NVIDIA identifies it as unsupported legacy software; the existing simulator is retained for reproducibility.
- S1 robot assets must be obtained with permission from their original owner. Supply the `yhrg_s1_description` directory, including the URDF and referenced meshes, through a read-only mount. The deployment does not copy the DyWA, S1 SDK, or shifu repositories.
- PyTorch, CUDA base images, FFmpeg, nv-codec-headers, MediaMTX, Python packages, and npm dependencies retain their respective licenses. Container builds obtain these dependencies from their upstream sources. Review those terms before redistributing built images.
- The browser robot preview assets and course materials have their own provenance and permissions. The source export includes application assets; the project owner must confirm redistribution rights before public upload.
- Runtime users, credentials, database contents, experimental videos, non-reference model checkpoints, and private environment files are excluded from the source export. The export is an allowlist, not a substitute for reviewing source files for accidentally embedded secrets.

Physical robot control remains disabled. Self-hosting includes the implemented learning, simulated training, evidence, evaluation, video, and teaching-agent features. UI capabilities marked planned or blocked are not converted into completed capabilities by deployment.

The small `deploy/reference-models.tar.gz` bundle contains exactly three project-generated initialization checkpoints identified in `deploy/reference-models.json`. These are required by the current Grasp, Lift and Lower/Release task contracts. Their redistribution is subject to the project owner confirming the applicable model and training-source permissions. No other training artifacts are bundled.

The browser S1 provenance file reports a CC BY-NC 4.0 declaration in the source DyWA repository, but the asset directory was untracked and its upstream history was not verified. This is a provenance note, not confirmation that every mesh/URDF may be redistributed or relicensed. Preserve applicable third-party notices and review container dependency terms before publishing images.
