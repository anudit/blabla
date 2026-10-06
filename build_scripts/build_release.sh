#!/bin/sh
set -eu

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
build_dir=${TMPDIR:-/tmp}/tts-metal-release-build
stage_dir=$(mktemp -d "$root/.BlaBla-release-staging.XXXXXX")
trap 'rm -rf "$stage_dir"' EXIT HUP INT TERM

xcodebuild \
    -project "$root/tts-metal/tts-metal.xcodeproj" \
    -scheme tts-metal \
    -configuration Release \
    -destination 'platform=macOS' \
    -derivedDataPath "$build_dir" \
    build CODE_SIGNING_ALLOWED=NO

source_app="$build_dir/Build/Products/Release/BlaBla.app"
staged_app="$stage_dir/BlaBla-Release.app"
output_app="$root/BlaBla-Release.app"
ditto "$source_app" "$staged_app"

# The target has ENABLE_APP_SANDBOX=NO. Keep the release signature aligned
# with that setting so saved history paths remain readable after relaunch.
codesign --force --deep --sign - "$staged_app"
codesign --verify --deep --strict "$staged_app"
if codesign -d --entitlements :- "$staged_app" 2>&1 | rg -q 'com.apple.security.app-sandbox'; then
    echo 'Release app unexpectedly has the app-sandbox entitlement' >&2
    exit 1
fi

if [ -e "$output_app" ]; then
    backup_dir=$(mktemp -d "${TMPDIR:-/tmp}/tts-metal-release-backup.XXXXXX")
    mv "$output_app" "$backup_dir/BlaBla-Release.app"
    echo "Previous app: $backup_dir/BlaBla-Release.app"
fi
mv "$staged_app" "$output_app"
echo "Release app: $output_app"
