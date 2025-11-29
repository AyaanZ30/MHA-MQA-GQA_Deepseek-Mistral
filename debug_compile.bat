@echo off

echo Step 1 : Compiling individual files
g++ -std=c++14 -c main.cpp 2>&1 | findstr /C:"error"
g++ -std=c++14 -c attention_common.cpp 2>&1 | findstr /C:"error"
g++ -std=c++14 -c mha.cpp 2>&1 | findstr /C:"error"
g++ -std=c++14 -c mqa.cpp 2>&1 | findstr /C:"error"
g++ -std=c++14 -c gqa.cpp 2>&1 | findstr /C:"error"

echo.

echo Step 2 : Linking all compiled files...
g++ -std=c++14 -o final.exe main.o attention_common.o mha.o mqa.o gqa.o -Wl,--verbose 2>&1

echo.

if exists final.exe(
    echo SUCCESS: final.exe created!
    echo.
    echo Running the program...
    echo ========================================
    final.exe
    pause
)else(
    echo FAILED : Could not create final.exe
    pause
)