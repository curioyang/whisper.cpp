#pragma once

#include <assert.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>
typedef unsigned char uint8;
typedef unsigned long uint32;
// uint32 base64_encode(char* input, uint8* encode);
int base64_decode(const uint8 *code, uint32 code_len, char *str);