#ifndef SIMCCL_EXTERNAL_SCHEDULE_H
#define SIMCCL_EXTERNAL_SCHEDULE_H

#include <cstdint>
#include <string>
#include <vector>

namespace MockNccl {

struct ExternalScheduleFlow {
  uint64_t flow_id;
  int src;
  int dst;
  uint64_t size_bytes;
  uint64_t start_ns;
};

class ExternalSchedule {
 public:
  static std::vector<ExternalScheduleFlow> load(
      const std::string& path,
      int num_ranks);
};

}  // namespace MockNccl

#endif  // SIMCCL_EXTERNAL_SCHEDULE_H
