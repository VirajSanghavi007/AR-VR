"""Shared pytest hooks."""
import http.server

import pytest

# Browser test files: each runs whole on one xdist worker (one Chrome per file, not one per worker).
ONE_WORKER = ("tests/test_site_page.py", "tests/test_vr_page.py", "tests/test_tagar_page.py", "tests/test_samples.py",
              "tests/test_scan_room.py", "tests/test_sketchup_sample.py")

# The page tests serve preview/ with ThreadingHTTPServer, whose listen backlog is 5. Chrome opens more module
# requests than that at once, and under load Windows refuses the overflow (ERR_CONNECTION_REFUSED), so a page's
# modules never ran. Tests only: this sets the class default for the test process.
http.server.ThreadingHTTPServer.request_queue_size = 128


def scope_of(nodeid):
    """The xdist scheduling unit of a test: the file for ONE_WORKER files (one worker, one Chrome), else the
    test itself (plain load balancing)."""
    path = nodeid.split("::", 1)[0]
    return path if path in ONE_WORKER else nodeid


@pytest.hookimpl(optionalhook=True)      # optional: runs without xdist too (-p no:xdist)
def pytest_xdist_make_scheduler(config, log):
    """20 software-rendered Chromes at once (one per worker) timed out on this laptop, so the Chrome page tests
    share a worker. xdist's own xdist_group marks do nothing on pytest 9 (it rewrites item._nodeid, which
    pytest 9 no longer reads), hence a scope scheduler."""
    if config.getvalue("dist") != "load":
        return None
    from xdist.scheduler import LoadScopeScheduling

    class OneWorkerFiles(LoadScopeScheduling):
        def _split_scope(self, nodeid):
            return scope_of(nodeid)

    return OneWorkerFiles(config, log)
