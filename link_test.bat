@echo off
echo Testing different linking approaches...

where g++
g++ --version

echo Approach 1: Link all .cpp files together
g++ -std=c++14 -o test1.exe main.cpp attention_common.cpp mha.cpp mqa.cpp gqa.cpp 2>&1

if %errorlevel% neq 0 (
    echo.
    echo Approach 1 failed, trying Approach 2...
    echo Approach 2: Link with verbose output
    g++ -std=c++14 -o test2.exe main.cpp attention_common.cpp mha.cpp mqa.cpp gqa.cpp -Wl,--verbose 2>&1 | findstr /C:"error:" /C:"undefined"
)

if exist test1.exe (
    echo SUCCESS: test1.exe created!
    test1.exe
    pause
) else if exist test2.exe (
    echo SUCCESS: test2.exe created!  
    test2.exe
    pause
) else (
    echo All approaches failed.
    pause
)