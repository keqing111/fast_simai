#include "ExternalSchedule.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace MockNccl {
namespace {

uint64_t parse_unsigned(
    const std::string& token,
    std::size_t line_number,
    const char* field) {
  if (token.empty() ||
      !std::all_of(token.begin(), token.end(), [](char value) {
        return value >= '0' && value <= '9';
      })) {
    throw std::runtime_error(
        "external schedule line " + std::to_string(line_number) +
        ": " + field + " must be an unsigned decimal integer");
  }
  try {
    std::size_t consumed = 0;
    const uint64_t value = std::stoull(token, &consumed, 10);
    if (consumed != token.size()) {
      throw std::runtime_error("trailing characters");
    }
    return value;
  } catch (const std::exception&) {
    throw std::runtime_error(
        "external schedule line " + std::to_string(line_number) +
        ": " + field + " is outside uint64 range");
  }
}

}  // namespace

std::vector<ExternalScheduleFlow> ExternalSchedule::load(
    const std::string& path,
    int num_ranks) {
  if (num_ranks <= 0) {
    throw std::runtime_error("external schedule requires a positive rank count");
  }
  std::ifstream input(path);
  if (!input.is_open()) {
    throw std::runtime_error("cannot open external schedule: " + path);
  }

  std::vector<ExternalScheduleFlow> flows;
  std::string line;
  std::size_t line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    if (line.empty()) {
      throw std::runtime_error(
          "external schedule line " + std::to_string(line_number) +
          ": empty rows are not allowed");
    }
    std::vector<std::string> columns;
    std::stringstream row(line);
    std::string token;
    while (std::getline(row, token, ',')) {
      columns.push_back(token);
    }
    if (!line.empty() && line.back() == ',') {
      columns.push_back("");
    }
    if (columns.size() != 4) {
      throw std::runtime_error(
          "external schedule line " + std::to_string(line_number) +
          ": expected exactly four columns");
    }

    const uint64_t src = parse_unsigned(columns[0], line_number, "src");
    const uint64_t dst = parse_unsigned(columns[1], line_number, "dst");
    const uint64_t size = parse_unsigned(columns[2], line_number, "size_bytes");
    const uint64_t start = parse_unsigned(columns[3], line_number, "start_ns");
    if (src > static_cast<uint64_t>(INT_MAX) ||
        dst > static_cast<uint64_t>(INT_MAX) ||
        src >= static_cast<uint64_t>(num_ranks) ||
        dst >= static_cast<uint64_t>(num_ranks)) {
      throw std::runtime_error(
          "external schedule line " + std::to_string(line_number) +
          ": endpoint is outside the rank range");
    }
    if (src == dst) {
      throw std::runtime_error(
          "external schedule line " + std::to_string(line_number) +
          ": self flows are not allowed");
    }
    if (size == 0) {
      throw std::runtime_error(
          "external schedule line " + std::to_string(line_number) +
          ": size_bytes must be positive");
    }
    flows.push_back(
        {static_cast<uint64_t>(flows.size()),
         static_cast<int>(src),
         static_cast<int>(dst),
         size,
         start});
  }
  if (flows.empty()) {
    throw std::runtime_error("external schedule is empty");
  }
  return flows;
}

}  // namespace MockNccl
