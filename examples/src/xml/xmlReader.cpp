// This file is part of RaiSim. You must obtain a valid license from RaiSim Tech
// Inc. prior to usage.

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"

int main(int argc, char* argv[]) {
  auto binaryPath = raisim::Path::setFromArgv(argv[0]);
  raisim::World::setActivationKey(binaryPath.getDirectory() + "\\rsc\\activation.raisim");

  RSFATAL_IF(argc != 2 || !strcmp("--help", argv[1]) || !strcmp("-h", argv[1]),
      "Requires the full path to the XML file as the first argument.")

  std::string xmlPath = argv[1];
  raisim::World world(xmlPath);
  raisim::RaisimServer server(&world);
  server.startRecordingVideo("heightMapUsingPng.mp4");

  server.launchServer();
  for (int i=0; i<10000; i++) {
    world.integrate();
    std::this_thread::sleep_for(std::chrono::milliseconds(size_t(1000 * world.getTimeStep())));
  }

  server.stopRecordingVideo();
  server.killServer();
}