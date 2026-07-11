#pragma once

#define MNW_SWITCH(COND1, COND2, COND3, NAME1, NAME2, NAME3, ...) \
  [&] {                                                           \
    if (COND1 == 16 && COND2 == 128 && COND3 == 1) {              \
      constexpr int NAME1 = 16;                                   \
      constexpr int NAME2 = 128;                                  \
      constexpr int NAME3 = 1;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 16 && COND2 == 64 && COND3 == 1) {        \
      constexpr int NAME1 = 16;                                   \
      constexpr int NAME2 = 64;                                   \
      constexpr int NAME3 = 1;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 16 && COND2 == 32 && COND3 == 1) {        \
      constexpr int NAME1 = 16;                                   \
      constexpr int NAME2 = 32;                                   \
      constexpr int NAME3 = 1;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 16 && COND2 == 16 && COND3 == 1) {        \
      constexpr int NAME1 = 16;                                   \
      constexpr int NAME2 = 16;                                   \
      constexpr int NAME3 = 1;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 64 && COND2 == 128 && COND3 == 4) {       \
      constexpr int NAME1 = 64;                                   \
      constexpr int NAME2 = 128;                                  \
      constexpr int NAME3 = 4;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 64 && COND2 == 64 && COND3 == 4) {        \
      constexpr int NAME1 = 64;                                   \
      constexpr int NAME2 = 64;                                   \
      constexpr int NAME3 = 4;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 64 && COND2 == 32 && COND3 == 4) {        \
      constexpr int NAME1 = 64;                                   \
      constexpr int NAME2 = 32;                                   \
      constexpr int NAME3 = 4;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 64 && COND2 == 16 && COND3 == 4) {        \
      constexpr int NAME1 = 64;                                   \
      constexpr int NAME2 = 16;                                   \
      constexpr int NAME3 = 4;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 32 && COND2 == 128 && COND3 == 2) {       \
      constexpr int NAME1 = 32;                                   \
      constexpr int NAME2 = 128;                                  \
      constexpr int NAME3 = 2;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 32 && COND2 == 64 && COND3 == 2) {        \
      constexpr int NAME1 = 32;                                   \
      constexpr int NAME2 = 64;                                   \
      constexpr int NAME3 = 2;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 32 && COND2 == 32 && COND3 == 2) {        \
      constexpr int NAME1 = 32;                                   \
      constexpr int NAME2 = 32;                                   \
      constexpr int NAME3 = 2;                                    \
      return __VA_ARGS__();                                       \
    } else if (COND1 == 32 && COND2 == 16 && COND3 == 2) {        \
      constexpr int NAME1 = 32;                                   \
      constexpr int NAME2 = 16;                                   \
      constexpr int NAME3 = 2;                                    \
      return __VA_ARGS__();                                       \
    }                                                             \
  }()

#define GBLOCKM_SWITCH(MAX_SPLITS, ...) \
  [&] {                                 \
    if (MAX_SPLITS <= 4) {              \
      constexpr static int BLOCKM = 4;  \
      return __VA_ARGS__();             \
    } else if (MAX_SPLITS <= 8) {       \
      constexpr static int BLOCKM = 8;  \
      return __VA_ARGS__();             \
    } else if (MAX_SPLITS <= 16) {      \
      constexpr static int BLOCKM = 16; \
      return __VA_ARGS__();             \
    } else if (MAX_SPLITS <= 32) {      \
      constexpr static int BLOCKM = 32; \
      return __VA_ARGS__();             \
    }                                   \
  }()
