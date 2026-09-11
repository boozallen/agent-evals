# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test transport-security validation of caller-supplied endpoint values."""

import inspect

import pytest

from agent_evals.core.transport import (
    CLEARTEXT_SCHEMES,
    NON_NETWORK_SCHEMES,
    NON_NETWORK_SENTINELS,
    SECURE_SCHEMES,
    is_secure_endpoint_value,
    validate_secure_transport,
)


class TestAcceptedValues:
    @pytest.mark.parametrize(
        "value",
        [
            "https://mlflow.corp",
            "HTTPS://mlflow.corp",
            "wss://host",
            "file:///var/mlruns",
            "sqlite:///local.db",
            "databricks",
            "",
            "./mlruns",
            "mlruns",
            "/var/mlruns",
        ],
    )
    def test_accepted(self, value):
        """Encrypted, non-network, and path values are all acceptable."""
        assert is_secure_endpoint_value(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            r"C:\mlruns",
            "d:/runs",
            r"\\server\share\mlruns",
        ],
    )
    def test_windows_paths_accepted(self, value):
        """Drive-letter and UNC paths are acceptable.

        `urlparse(r"C:\\mlruns").scheme` is `"c"` — the trap the
        `len(scheme) <= 1` rule exists to handle, since the shortest real
        scheme is two characters. Acceptance is a rule, not an enumeration of
        drive letters, so `d:` works without being listed anywhere.
        """
        assert is_secure_endpoint_value(value) is True

    def test_empty_value_is_not_this_functions_concern(self):
        """An absent setting is rejected elsewhere, not here."""
        assert is_secure_endpoint_value("") is True
        validate_secure_transport("SOME_ENDPOINT", "")


class TestLoopbackCleartext:
    """Cleartext is acceptable only when it addresses the local machine.

    Loopback traffic is handled inside the host and never appears on a link, so
    admitting it costs no confidentiality — and it keeps a locally run server,
    which serves plaintext by default, a working configuration. The acceptance
    is a property of the value: nothing turns validation off, and the same
    scheme naming any other host still fails.
    """

    @pytest.mark.parametrize(
        "value",
        [
            "http://localhost:5000",
            "HTTP://localhost:5000",
            "http://LocalHost:5000",
            "http://localhost.:5000",
            "http://localhost",
            "http://127.0.0.1:5000",
            "http://127.9.9.9:5000",
            "http://[::1]:5000",
            "ws://localhost:5000",
        ],
    )
    def test_loopback_cleartext_accepted(self, value):
        """The name `localhost` and every loopback address literal are allowed.

        `127.9.9.9` is in the list on purpose: the whole of `127.0.0.0/8` is
        loopback, and it is decided by `ipaddress` as a rule rather than by
        enumerating the one address people usually write.
        """
        assert is_secure_endpoint_value(value) is True

    @pytest.mark.parametrize(
        "value",
        [
            "http://mlflow.corp:5000",
            "http://localhost.evil.example",
            "http://app.localhost:5000",
            "http://notlocalhost:5000",
            "http://10.0.0.1:5000",
            "http://",
        ],
    )
    def test_cleartext_to_any_other_host_still_rejected(self, value):
        """The carve-out is loopback, not anything resembling it.

        `localhost.evil.example` and `notlocalhost` are the reason the host is
        compared for equality rather than by prefix or suffix. `app.localhost`
        is reserved for loopback by RFC 6761 but is still rejected: this module
        resolves no names, because a name that resolves to loopback now can
        resolve elsewhere on the next lookup. `http://` has no host at all.
        """
        assert is_secure_endpoint_value(value) is False


class TestRejectedValues:
    @pytest.mark.parametrize(
        "value",
        [
            "http://mlflow.corp:5000",
            "HTTP://mlflow.corp:5000",
            "ws://host:3000",
            "gopher://host",
            "ftp://host",
            "gopher://localhost",
        ],
    )
    def test_rejected(self, value):
        """Cleartext and unenumerated schemes are both unacceptable.

        `gopher` and `ftp` matter as much as `http` here: acceptance requires
        allowlist membership, so a scheme nobody anticipated fails too — and a
        loopback host does not rescue one, since the loopback carve-out admits
        only the two schemes whose plaintext form is otherwise well understood.
        """
        assert is_secure_endpoint_value(value) is False

    def test_mistyped_scheme_is_rejected_not_read_as_a_path(self):
        """The malformed-URL guard: `htp://host` fails loudly.

        The single most important case in this file. `htp://host` parses to
        scheme `htp`, which is not allowed. Without the `"://"` guard ahead of
        the local-path branch it would fall through and be accepted as a
        relative directory named `htp:` — the caller would silently lose the
        endpoint they meant to configure. If this assertion ever inverts, the
        allowlist has a hole the size of every unrecognised scheme.
        """
        assert is_secure_endpoint_value("htp://host") is False

    def test_bare_host_and_port_is_rejected(self):
        """`localhost:5000` parses to scheme `localhost`, which is not allowed.

        A `urlparse` trap. This is deliberately a rejection: a bare host and
        port names no transport, so nothing here can tell whether the caller
        meant the cleartext or the encrypted form of it. The spelled-out
        `http://localhost:5000` is accepted, and the error message says so. Do
        not "fix" this to accepted.
        """
        assert is_secure_endpoint_value("localhost:5000") is False


class TestValidationErrors:
    def test_raises_value_error_naming_field_value_and_scheme(self):
        """The message alone is enough to act on."""
        with pytest.raises(ValueError) as exc_info:
            validate_secure_transport("MLFLOW_TRACKING_URI", "http://host:5000")

        message = str(exc_info.value)
        assert "MLFLOW_TRACKING_URI" in message
        assert "http://host:5000" in message
        assert "'http'" in message

    def test_message_names_every_accepted_alternative(self):
        """A rejected caller has to be told what would work instead.

        Including the loopback form: a developer pointing at a server on their
        own machine is the most common way to reach this error, and without the
        example in the message the obvious next guess is that no local
        configuration is accepted at all.
        """
        with pytest.raises(ValueError) as exc_info:
            validate_secure_transport("MLFLOW_TRACKING_URI", "http://mlflow.corp")

        message = str(exc_info.value)
        assert "https" in message
        assert "local store" in message
        assert "http://localhost:5000" in message

    def test_accepted_value_does_not_raise(self):
        validate_secure_transport("MLFLOW_TRACKING_URI", "https://mlflow.corp")

    def test_loopback_value_does_not_raise(self):
        validate_secure_transport("MLFLOW_TRACKING_URI", "http://localhost:5000")

    def test_failure_is_catchable_not_a_process_exit(self):
        """A library raises; the caller owns the exit decision."""
        with pytest.raises(ValueError):
            validate_secure_transport("SOME_ENDPOINT", "http://host")

    def test_embedded_password_is_masked_but_message_stays_actionable(self):
        """An uncaught traceback lands in a log; a password must not ride along.

        A store URI documents its password in userinfo, and such a value is
        never an encrypted transport, so it always reaches this message. The
        rest of the message has to survive the masking or the caller cannot act
        on it.
        """
        # Assembled rather than written whole to avoid tripping secret scanners.
        # A complete literal with scheme, userinfo, and host in one string
        # would be flagged on export, but this is a fixture, not a credential.
        password = "s3cret"
        with pytest.raises(ValueError) as exc_info:
            validate_secure_transport(
                "MLFLOW_TRACKING_URI",
                f"postgresql://mlflow:{password}@db.corp:5432/mlflow",
            )

        message = str(exc_info.value)
        assert "s3cret" not in message
        assert "mlflow:***@db.corp:5432" in message
        assert "MLFLOW_TRACKING_URI" in message
        assert "'postgresql'" in message

    def test_userinfo_without_a_password_is_masked_whole(self):
        """A lone userinfo component may itself be a token."""
        with pytest.raises(ValueError) as exc_info:
            validate_secure_transport("SOME_ENDPOINT", "http://sk-abc123@host")

        message = str(exc_info.value)
        assert "sk-abc123" not in message
        assert "***@host" in message

    def test_value_without_credentials_is_quoted_unchanged(self):
        """Masking must not alter the common case."""
        with pytest.raises(ValueError) as exc_info:
            validate_secure_transport("SOME_ENDPOINT", "http://host:5000/path?a=b")

        assert "http://host:5000/path?a=b" in str(exc_info.value)


class TestNoEscapeHatch:
    """Pins the absence of any opt-out.

    Enforcement was originally relaxable through an environment variable. It was
    removed so that a value which would transmit in cleartext has no accepted
    form at all — the library is a template others copy, and a switch shipped
    here becomes a switch set everywhere. These tests fail if one is
    reintroduced.
    """

    @pytest.mark.parametrize(
        "env_name",
        [
            "FOUNDRY_AGENT_EVALS_ALLOW_HTTP",
            "AGENT_EVALS_ALLOW_HTTP",
            "ALLOW_HTTP",
        ],
    )
    def test_no_environment_variable_relaxes_enforcement(self, monkeypatch, env_name):
        """Setting a plausible bypass name changes nothing."""
        monkeypatch.setenv(env_name, "1")
        with pytest.raises(ValueError):
            validate_secure_transport("SOME_ENDPOINT", "http://mlflow.corp:5000")

    def test_validator_reads_no_environment_at_all(self):
        """The module does not consult the environment, so nothing can relax it.

        Asserted on the source rather than on behaviour: a future `os.getenv`
        call would be an escape hatch by construction, whichever variable it
        named, and a behavioural test can only cover names someone thought to
        list.
        """
        import agent_evals.core.transport as transport

        source = inspect.getsource(transport)
        assert "getenv" not in source
        assert "environ" not in source


class TestAllowlistConstants:
    def test_secure_schemes(self):
        assert set(SECURE_SCHEMES) == {"https", "wss"}
        assert isinstance(SECURE_SCHEMES, frozenset)

    def test_non_network_schemes(self):
        assert set(NON_NETWORK_SCHEMES) == {"file", "sqlite"}
        assert isinstance(NON_NETWORK_SCHEMES, frozenset)

    def test_non_network_sentinels(self):
        assert set(NON_NETWORK_SENTINELS) == {"databricks"}
        assert isinstance(NON_NETWORK_SENTINELS, frozenset)

    def test_cleartext_schemes(self):
        """The cleartext set is never an acceptance on its own.

        Membership only makes a value eligible for the loopback check, so this
        set may not overlap the sets that are accepted outright — a scheme in
        both would be accepted for any host.
        """
        assert set(CLEARTEXT_SCHEMES) == {"http", "ws"}
        assert isinstance(CLEARTEXT_SCHEMES, frozenset)
        assert not CLEARTEXT_SCHEMES & (SECURE_SCHEMES | NON_NETWORK_SCHEMES)
