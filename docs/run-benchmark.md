## Benchmarks

### DEBUG QuixBugs
```bash
javac -cp .:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar java_testcases/junit/GCD_TEST.java
java -cp .:java_programs:junit4-4.12.jar:hamcrest-all-1.3.jar org.junit.runner.JUnitCore java_testcases.junit.GCD_TEST
```

### DEBUG defects4j
```bash
defects4j checkout -p Chart -v 4b -w /home/yglee/wl/p14/nosync/defects4j_tmp853/tmp
```

```bash
# --do_humaneval
# --do_quixbugs
# --do_defects4j --strict_defects4j --validate_result_split_defects4j

# --do_generate
# --do_validate
```
