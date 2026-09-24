// Generator-facing CLI only. Extracted from jdnitrap/bitmood.
#pragma once

namespace cmix {

int cmd_train(int argc, char** argv, int start);
int cmd_generate(int argc, char** argv, int start);
int cmd_info(int argc, char** argv, int start);
int cmd_write(int argc, char** argv, int start);
int cmd_graph(int argc, char** argv, int start);

}  // namespace cmix
