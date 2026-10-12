#include "ExternalSchedule.h"

#include <cassert>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

using MockNccl::ExternalSchedule;
using MockNccl::ExternalScheduleFlow;

static std::string write_temp(const std::string& contents) {
  char path[] = "/tmp/simai-external-schedule-XXXXXX";
  int fd = mkstemp(path);
  assert(fd >= 0);
  close(fd);
  std::ofstream output(path);
  output << contents;
  output.close();
  return path;
}

static void expect_failure(
    const std::string& contents,
    const std::string& expected_text,
    int ranks = 2) {
  const std::string path = write_temp(contents);
  bool failed = false;
  try {
    ExternalSchedule::load(path, ranks);
  } catch (const std::runtime_error& error) {
    failed = true;
    assert(std::string(error.what()).find(expected_text) != std::string::npos);
  }
  std::remove(path.c_str());
  assert(failed);
}

int main() {
  const std::string valid_path = write_temp("0,1,64,0\n0,1,64,10\n1,0,7,10\n");
  const std::vector<ExternalScheduleFlow> flows = ExternalSchedule::load(valid_path, 2);
  std::remove(valid_path.c_str());
  assert(flows.size() == 3);
  for (std::size_t index = 0; index < flows.size(); ++index) {
    assert(flows[index].flow_id == index);
  }
  assert(flows[0].src == 0 && flows[0].dst == 1);
  assert(flows[0].size_bytes == 64 && flows[0].start_ns == 0);
  assert(flows[1].src == 0 && flows[1].dst == 1);
  assert(flows[1].size_bytes == 64 && flows[1].start_ns == 10);

  expect_failure("", "empty");
  expect_failure("src,dst,size,start\n", "line 1");
  expect_failure("0,1,64\n", "line 1");
  expect_failure("0,1,64,0,extra\n", "line 1");
  expect_failure("0, ,64,0\n", "line 1");
  expect_failure("0,0,64,0\n", "line 1");
  expect_failure("0,1,0,0\n", "line 1");
  expect_failure("-1,1,64,0\n", "line 1");
  expect_failure("0,1,64,-1\n", "line 1");
  expect_failure("0,1,18446744073709551616,0\n", "line 1");
  expect_failure("0,2,64,0\n", "line 1");
  expect_failure("0,1,64,0\n\n", "line 2");
  return 0;
}
