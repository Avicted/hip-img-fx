#pragma once

/**
 * @brief Main application entry point (testable interface)
 *
 * This is the actual application logic extracted from main() to make it testable.
 * The real main() in main.cpp just calls this function.
 *
 * @param argc Argument count
 * @param argv Argument values
 * @return Exit code (0 for success, non-zero for failure)
 */
int app_main(int argc, char **argv);
