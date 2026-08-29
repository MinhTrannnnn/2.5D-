#!/bin/sh
set -eu

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
build_dir=$(mktemp -d /private/tmp/figure1-options.XXXXXX)
module_cache="$build_dir/module-cache"
mkdir -p "$module_cache"

CLANG_MODULE_CACHE_PATH="$module_cache" clang \
  -fobjc-arc \
  -framework AppKit \
  "$script_dir/make_figure1_options.m" \
  -o "$build_dir/make_figure1_options"

"$build_dir/make_figure1_options" "$script_dir"
