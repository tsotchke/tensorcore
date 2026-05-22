# native_sdk_consumer

Standalone CMake consumer for an installed tensorcore native SDK. This is
the fixture used by release smoke tests and SDK archive verification.

```sh
cmake -S examples/native_sdk_consumer -B /tmp/tc-consumer \
  -DCMAKE_PREFIX_PATH=/opt/tensorcore
cmake --build /tmp/tc-consumer
DYLD_LIBRARY_PATH=/opt/tensorcore/lib /tmp/tc-consumer/consumer_shared
/tmp/tc-consumer/consumer_static
DYLD_LIBRARY_PATH=/opt/tensorcore/lib /tmp/tc-consumer/consumer_cxx
```

The default executables exercise public ABI helpers without requiring a
real GPU, so they are safe on GitHub-hosted macOS runners. Set
`TC_CONSUMER_RUN_INIT=1` when you also want the C consumer to prove that
the installed runtime initializes on the current host.
