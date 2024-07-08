### run_tests.py ###
# author: Elliott Walker
# last update: 8 July 2024
# description: runs all tests in tests directory; see info in docs/tests.txt
# usage: python run_tests.py

def RUN_ALL_TESTS():
    import os
    import context
    tests = {}
    for module in os.listdir(os.path.dirname(__file__)):
        if module[-3:] == '.py' and module [:5] == 'TEST-':
            MOD = __import__(module[:-3], locals(), globals())
            tests[module[5:-3]] = getattr(MOD, 'RUN_TESTS')
    for module, test in tests.items():
        code = test()
        assert code == 0, f"Failure in module '{module}' - status code {code}"

if __name__ == '__main__':
    RUN_ALL_TESTS()
