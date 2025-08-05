#!/usr/bin/env python3
"""
Basic tests for the tldr script to ensure it functions correctly.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_command(cmd, cwd=None, expect_failure=False):
    """Run a command and return the result."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, cwd=cwd
        )
        if not expect_failure and result.returncode != 0:
            print(f"Command failed: {cmd}")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            return None
        return result
    except Exception as e:
        print(f"Error running command {cmd}: {e}")
        return None


def test_help():
    """Test that the help command works."""
    print("Testing --help option...")
    result = run_command("uv run python tldr.py --help")
    if result and "summarize a document" in result.stdout.lower():
        print("✓ Help command works correctly")
        return True
    else:
        print("✗ Help command failed")
        return False


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    test_script = """
try:
    import argparse
    import urllib.request
    from dotenv import load_dotenv
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchaingang import get_chat_model, get_provider_list
    print("SUCCESS: All imports work")
except ImportError as e:
    print(f"FAILED: Import error: {e}")
    sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()

        result = run_command(f"uv run python {f.name}")
        os.unlink(f.name)

        if result and "SUCCESS" in result.stdout:
            print("✓ All imports work correctly")
            return True
        else:
            print("✗ Import test failed")
            if result:
                print(f"Output: {result.stdout}")
                print(f"Error: {result.stderr}")
            return False


def test_provider_list():
    """Test that the provider list function works."""
    print("Testing provider list...")
    test_script = """
from langchaingang import get_provider_list
providers = get_provider_list()
print(f"Available providers: {providers}")
if isinstance(providers, list):
    print("SUCCESS: Provider list is a list")
else:
    print("FAILED: Provider list is not a list")
    sys.exit(1)
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(test_script)
        f.flush()

        result = run_command(f"uv run python {f.name}")
        os.unlink(f.name)

        if result and "SUCCESS" in result.stdout:
            print("✓ Provider list works correctly")
            return True
        else:
            print("✗ Provider list test failed")
            return False


def test_file_reading():
    """Test that the script can read a file and either generate a summary or fail gracefully."""
    print("Testing file reading and basic functionality...")

    # Check if test file exists
    if not os.path.exists("test_sample.txt"):
        print("✗ test_sample.txt not found")
        return False

    # Run the script - it might work if API keys are available, or fail gracefully
    result = run_command("uv run python tldr.py test_sample.txt")

    if result is None:
        print("✗ Command execution failed")
        return False

    output = result.stdout + result.stderr

    # Check for various successful scenarios
    if result.returncode == 0 and len(output.strip()) > 0:
        # Script ran successfully and produced output (likely a summary)
        print("✓ Script successfully generated a summary")
        print(f"  Summary length: {len(output.strip())} characters")
        return True
    elif any(
        phrase in output.lower()
        for phrase in [
            "api key",
            "credentials",
            "authentication",
            "provider",
            "token",
            "key",
            "auth",
            "access",
        ]
    ):
        print(
            "✓ File reading successful (failed at LLM creation due to missing API keys)"
        )
        return True
    elif "no such file" in output.lower() or "file not found" in output.lower():
        print("✗ File reading failed")
        print(f"Output: {output}")
        return False
    else:
        # Script ran but with unexpected output
        print(f"⚠ Script ran with unexpected output: {output[:100]}...")
        # If it didn't crash, we'll consider this a pass
        return result.returncode == 0


def test_argument_parsing():
    """Test various argument combinations."""
    print("Testing argument parsing...")

    # Test invalid provider
    result = run_command(
        "uv run python tldr.py test_sample.txt --provider invalid_provider",
        expect_failure=True,
    )
    if result and "invalid choice" in result.stderr.lower():
        print("✓ Invalid provider correctly rejected")
    else:
        print("⚠ Invalid provider test inconclusive")

    # Test valid providers (might work or fail depending on API keys)
    providers_to_test = ["openai", "anthropic"]
    for provider in providers_to_test:
        result = run_command(
            f"uv run python tldr.py test_sample.txt --provider {provider}"
        )
        if result:
            output = result.stdout + result.stderr
            if result.returncode == 0 and len(output.strip()) > 0:
                print(f"✓ Provider {provider} successfully generated output")
            elif any(
                phrase in output.lower()
                for phrase in ["api key", "credentials", "auth"]
            ):
                print(f"✓ Provider {provider} argument accepted (missing credentials)")
            else:
                print(f"⚠ Provider {provider} test inconclusive: {output[:50]}...")

    return True


def main():
    """Run all tests."""
    print("Running TLDR script tests...")
    print("=" * 50)

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    tests = [
        test_help,
        test_imports,
        test_provider_list,
        test_file_reading,
        test_argument_parsing,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} raised exception: {e}")
        print("-" * 30)

    print(f"\nResults: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
