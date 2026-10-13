# Load-test scenarios

Run `jmeter -n -t ragflow_load_test.jmx -JbaseUrl=http://localhost:9380`. Extend the plan with authenticated chat and upload requests. Record P95 latency, error rate, Redis queue depth, container RSS, and GPU memory for baseline and optimized runs.
