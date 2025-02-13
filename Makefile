# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Default target executed when no arguments are given to make.
default_target: all

.PHONY : default_target

# Allow only one "make -f Makefile2" at a time, but pass parallelism.
.NOTPARALLEL:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/xuesong/RailInspection

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/xuesong/RailInspection

#=============================================================================
# Targets provided globally by CMake.

# Special rule for the target edit_cache
edit_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "No interactive CMake dialog available..."
	/usr/bin/cmake -E echo No\ interactive\ CMake\ dialog\ available.
.PHONY : edit_cache

# Special rule for the target edit_cache
edit_cache/fast: edit_cache

.PHONY : edit_cache/fast

# Special rule for the target rebuild_cache
rebuild_cache:
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --cyan "Running CMake to regenerate build system..."
	/usr/bin/cmake -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR)
.PHONY : rebuild_cache

# Special rule for the target rebuild_cache
rebuild_cache/fast: rebuild_cache

.PHONY : rebuild_cache/fast

# The main all target
all: cmake_check_build_system
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xuesong/RailInspection/CMakeFiles /home/xuesong/RailInspection/CMakeFiles/progress.marks
	$(MAKE) -f CMakeFiles/Makefile2 all
	$(CMAKE_COMMAND) -E cmake_progress_start /home/xuesong/RailInspection/CMakeFiles 0
.PHONY : all

# The main clean target
clean:
	$(MAKE) -f CMakeFiles/Makefile2 clean
.PHONY : clean

# The main clean target
clean/fast: clean

.PHONY : clean/fast

# Prepare targets for installation.
preinstall: all
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall

# Prepare targets for installation.
preinstall/fast:
	$(MAKE) -f CMakeFiles/Makefile2 preinstall
.PHONY : preinstall/fast

# clear depends
depend:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 1
.PHONY : depend

#=============================================================================
# Target rules for targets named surface_defects

# Build rule for target.
surface_defects: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 surface_defects
.PHONY : surface_defects

# fast build rule for target.
surface_defects/fast:
	$(MAKE) -f CMakeFiles/surface_defects.dir/build.make CMakeFiles/surface_defects.dir/build
.PHONY : surface_defects/fast

#=============================================================================
# Target rules for targets named yulin

# Build rule for target.
yulin: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 yulin
.PHONY : yulin

# fast build rule for target.
yulin/fast:
	$(MAKE) -f CMakeFiles/yulin.dir/build.make CMakeFiles/yulin.dir/build
.PHONY : yulin/fast

#=============================================================================
# Target rules for targets named defects

# Build rule for target.
defects: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 defects
.PHONY : defects

# fast build rule for target.
defects/fast:
	$(MAKE) -f CMakeFiles/defects.dir/build.make CMakeFiles/defects.dir/build
.PHONY : defects/fast

#=============================================================================
# Target rules for targets named test

# Build rule for target.
test: cmake_check_build_system
	$(MAKE) -f CMakeFiles/Makefile2 test
.PHONY : test

# fast build rule for target.
test/fast:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/build
.PHONY : test/fast

defectsDetect.o: defectsDetect.cpp.o

.PHONY : defectsDetect.o

# target to build an object file
defectsDetect.cpp.o:
	$(MAKE) -f CMakeFiles/defects.dir/build.make CMakeFiles/defects.dir/defectsDetect.cpp.o
.PHONY : defectsDetect.cpp.o

defectsDetect.i: defectsDetect.cpp.i

.PHONY : defectsDetect.i

# target to preprocess a source file
defectsDetect.cpp.i:
	$(MAKE) -f CMakeFiles/defects.dir/build.make CMakeFiles/defects.dir/defectsDetect.cpp.i
.PHONY : defectsDetect.cpp.i

defectsDetect.s: defectsDetect.cpp.s

.PHONY : defectsDetect.s

# target to generate assembly for a file
defectsDetect.cpp.s:
	$(MAKE) -f CMakeFiles/defects.dir/build.make CMakeFiles/defects.dir/defectsDetect.cpp.s
.PHONY : defectsDetect.cpp.s

surfaceDefects.o: surfaceDefects.cpp.o

.PHONY : surfaceDefects.o

# target to build an object file
surfaceDefects.cpp.o:
	$(MAKE) -f CMakeFiles/surface_defects.dir/build.make CMakeFiles/surface_defects.dir/surfaceDefects.cpp.o
.PHONY : surfaceDefects.cpp.o

surfaceDefects.i: surfaceDefects.cpp.i

.PHONY : surfaceDefects.i

# target to preprocess a source file
surfaceDefects.cpp.i:
	$(MAKE) -f CMakeFiles/surface_defects.dir/build.make CMakeFiles/surface_defects.dir/surfaceDefects.cpp.i
.PHONY : surfaceDefects.cpp.i

surfaceDefects.s: surfaceDefects.cpp.s

.PHONY : surfaceDefects.s

# target to generate assembly for a file
surfaceDefects.cpp.s:
	$(MAKE) -f CMakeFiles/surface_defects.dir/build.make CMakeFiles/surface_defects.dir/surfaceDefects.cpp.s
.PHONY : surfaceDefects.cpp.s

testFunc.o: testFunc.cpp.o

.PHONY : testFunc.o

# target to build an object file
testFunc.cpp.o:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/testFunc.cpp.o
.PHONY : testFunc.cpp.o

testFunc.i: testFunc.cpp.i

.PHONY : testFunc.i

# target to preprocess a source file
testFunc.cpp.i:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/testFunc.cpp.i
.PHONY : testFunc.cpp.i

testFunc.s: testFunc.cpp.s

.PHONY : testFunc.s

# target to generate assembly for a file
testFunc.cpp.s:
	$(MAKE) -f CMakeFiles/test.dir/build.make CMakeFiles/test.dir/testFunc.cpp.s
.PHONY : testFunc.cpp.s

yulinshang.o: yulinshang.cpp.o

.PHONY : yulinshang.o

# target to build an object file
yulinshang.cpp.o:
	$(MAKE) -f CMakeFiles/yulin.dir/build.make CMakeFiles/yulin.dir/yulinshang.cpp.o
.PHONY : yulinshang.cpp.o

yulinshang.i: yulinshang.cpp.i

.PHONY : yulinshang.i

# target to preprocess a source file
yulinshang.cpp.i:
	$(MAKE) -f CMakeFiles/yulin.dir/build.make CMakeFiles/yulin.dir/yulinshang.cpp.i
.PHONY : yulinshang.cpp.i

yulinshang.s: yulinshang.cpp.s

.PHONY : yulinshang.s

# target to generate assembly for a file
yulinshang.cpp.s:
	$(MAKE) -f CMakeFiles/yulin.dir/build.make CMakeFiles/yulin.dir/yulinshang.cpp.s
.PHONY : yulinshang.cpp.s

# Help Target
help:
	@echo "The following are some of the valid targets for this Makefile:"
	@echo "... all (the default if no target is provided)"
	@echo "... clean"
	@echo "... depend"
	@echo "... edit_cache"
	@echo "... rebuild_cache"
	@echo "... surface_defects"
	@echo "... yulin"
	@echo "... defects"
	@echo "... test"
	@echo "... defectsDetect.o"
	@echo "... defectsDetect.i"
	@echo "... defectsDetect.s"
	@echo "... surfaceDefects.o"
	@echo "... surfaceDefects.i"
	@echo "... surfaceDefects.s"
	@echo "... testFunc.o"
	@echo "... testFunc.i"
	@echo "... testFunc.s"
	@echo "... yulinshang.o"
	@echo "... yulinshang.i"
	@echo "... yulinshang.s"
.PHONY : help



#=============================================================================
# Special targets to cleanup operation of make.

# Special rule to run CMake to check the build system integrity.
# No rule that depends on this can have commands that come from listfiles
# because they might be regenerated.
cmake_check_build_system:
	$(CMAKE_COMMAND) -H$(CMAKE_SOURCE_DIR) -B$(CMAKE_BINARY_DIR) --check-build-system CMakeFiles/Makefile.cmake 0
.PHONY : cmake_check_build_system

