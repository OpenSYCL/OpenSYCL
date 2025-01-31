/*
 * This file is part of AdaptiveCpp, an implementation of SYCL and C++ standard
 * parallelism for CPUs and GPUs.
 *
 * Copyright The AdaptiveCpp Contributors
 *
 * AdaptiveCpp is released under the BSD 2-Clause "Simplified" License.
 * See file LICENSE in the project root for full license details.
 */
// SPDX-License-Identifier: BSD-2-Clause

/// This is the AdaptiveCpp PCUDA preprocessor. This program should only be invoked 
/// on already preprocessed files (e.g. clang -E) prior to C++ parsing.

#include <iostream>
#include <fstream>

void help() {
  std::cout << "Usage: acpp-pcuda-pp <inputfile> <outputfile>" << std::endl;
}

bool read_file(const std::string& filename, std::string& out) {
  std::ifstream file{filename, std::ios::binary|std::ios::ate};
  if(!file.is_open())
    return false;

  auto size = file.tellg();

  if (size == 0) {
      out = std::string{};
      return true;
  }

  std::string result(size, '\0');

  file.seekg(0, std::ios::beg);
  file.read(result.data(), size);

  out = result;

  return true;
}


int main(int argc, char** argv) {
  if(argc != 3) {
    help();
    return -1;
  }

  std::string input_file = argv[1];
  std::string output_file = argv[2];

  std::string content;
  if(!read_file(input_file, content)) {
    std::cerr << "Could not read file: " << input_file << std::endl;
    return -1;
  }

  std::size_t pos = std::string::npos;
  
  while((pos = content.find("<<<")) !=  std::string::npos) {
    content.erase(pos, 3);
    content.insert(pos, "__pcudaPushCallConfiguration(");
    
    auto paren_close = content.find(">>>", pos);

    if(paren_close == std::string::npos) {
      std::cerr << "Detected invalid call configuration" << std::endl;
      return -1;
    } else {
      content.erase(paren_close, 3);
      // TODO A better transformation would be something like
      // (__pcudaPushCallConfiguration(...), kernel(....))
      // so that it remains a single statement.
      content.insert(paren_close, ");");
    }
  }

  std::ofstream out_file{output_file, std::ios::trunc};
  out_file.write(content.data(), content.size());

  return 0;
}
