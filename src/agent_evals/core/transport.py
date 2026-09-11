# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Transport-security validation for caller-supplied endpoint values.

A caller-supplied endpoint value is a string a caller provides to name where
data should be sent or stored. This module decides whether such a value is
acceptable, and raises when it is not, so that a value which would put data on a
network in cleartext is rejected before anything can open a connection with it.

The decision is an allowlist: a value is acceptable only if it names an
encrypted transport, or names no transport that leaves the machine it runs on.
Everything else is rejected, including a scheme nobody enumerated. This module
imports only from the standard library and performs no I/O.
"""

import ipaddress
import ntpath
from urllib.parse import urlparse, urlsplit, urlunsplit

# The four sets below classify a value's scheme, and together with the local-path
# branch they form an allowlist, not a denylist: a value is accepted only by
# matching one of them (see `is_secure_endpoint_value`). Deny-by-default is
# deliberate — a denylist only blocks what someone thought to enumerate, so an
# unanticipated scheme would pass silently.
#
# Three of the four admit a value on their own. `CLEARTEXT_SCHEMES` is the
# exception: matching it is a precondition, not an acceptance, and the value is
# admitted only if the host is also loopback.

#: Schemes that encrypt in transit.
SECURE_SCHEMES = frozenset({"https", "wss"})

#: Schemes that address a local store and transmit nothing over a network.
#: These are allowed because a local store transmits nothing, so admitting it
#: costs no confidentiality — and refusing it would leave a caller who wants no
#: server at all with no accepted value.
NON_NETWORK_SCHEMES = frozenset({"file", "sqlite"})

#: Schemes that transmit in cleartext. Accepted only when the host is the
#: machine the caller is running on, where the traffic never reaches a network.
#: Paired with `_is_loopback_host` — neither is a decision on its own.
CLEARTEXT_SCHEMES = frozenset({"http", "ws"})

#: Bare-word values that name a store to resolve from ambient configuration
#: rather than a URI. Matched exactly, not parsed.
NON_NETWORK_SENTINELS = frozenset({"databricks"})

_SCHEME_SEPARATOR = "://"

#: The one hostname treated as loopback by name. Every other loopback host must
#: be written as an address literal, which `ipaddress` can decide on its own.
_LOOPBACK_HOSTNAME = "localhost"


def _is_loopback_host(value: str) -> bool:
    """Report whether a URL's host is the machine the caller is running on.

    Accepts the name ``localhost`` and any loopback address literal — the whole
    of ``127.0.0.0/8`` and IPv6 ``::1`` — because `ipaddress` decides the
    literals as a rule rather than an enumeration.

    A hostname is otherwise not resolved: this module performs no I/O, and a
    name that resolves to a loopback address today can resolve elsewhere on the
    next lookup, so a resolved answer would not stay true. Names such as
    ``app.localhost``, which RFC 6761 reserves for loopback, are therefore
    rejected too; write the address literal instead.

    Args:
        value: The endpoint value to inspect.

    Returns:
        True when the value's host is loopback; False when it is any other host,
        or when the value has no host at all.
    """
    host = urlparse(value).hostname
    if not host:
        return False

    # `urlparse` lowercases the host. A trailing dot is the fully qualified
    # spelling of the same name, so `localhost.` is `localhost`.
    host = host.rstrip(".")
    if host == _LOOPBACK_HOSTNAME:
        return True

    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # Not an address literal, so it is a name this module will not resolve.
        return False


def is_secure_endpoint_value(value: str) -> bool:
    """Report whether a caller-supplied endpoint value is acceptable.

    Args:
        value: The endpoint value to classify.

    Returns:
        True when the value names an encrypted transport, names no transport at
        all, or addresses the machine it runs on in cleartext; False otherwise.
    """
    # An absent value is not this function's concern. Rejecting a setting that
    # was never provided is a separate, pre-existing responsibility of whoever
    # resolved it.
    if not value:
        return True

    if value in NON_NETWORK_SENTINELS:
        return True

    # `urlparse` lowercases the scheme, so no manual normalization is needed:
    # `HTTPS://host` and `https://host` both parse to `https`.
    scheme = urlparse(value).scheme

    if scheme in SECURE_SCHEMES:
        return True

    if scheme in NON_NETWORK_SCHEMES:
        return True

    # Cleartext to the local machine reaches no network: loopback traffic is
    # handled inside the host and never appears on a link where it could be
    # observed, so admitting it costs no confidentiality. This keeps a locally
    # run server — which serves plaintext by default and would otherwise need a
    # certificate or a TLS proxy to be usable at all — a working configuration.
    #
    # This is a property of the value, not a switch: nothing turns validation
    # off, and the same scheme naming any other host is still rejected below.
    if scheme in CLEARTEXT_SCHEMES and _is_loopback_host(value):
        return True

    # Load-bearing, and it must precede the local-path branch below. A value
    # that looks like a URL but whose scheme is not allowed is rejected here
    # rather than falling through. Without this, a mistyped scheme such as
    # `htp://host` parses to scheme `htp`, reaches the path branch, and is
    # silently accepted as a relative directory named `htp:` — the caller
    # loses the endpoint they meant to configure and gets no error. It also
    # keeps the path branch from turning every unrecognised scheme into an
    # accepted value, which would give this allowlist the failure mode of a
    # denylist.
    if _SCHEME_SEPARATOR in value:
        return False

    # A local filesystem path transmits nothing, so it is acceptable.
    #
    # `len(scheme) <= 1` is the Windows drive-letter carve-out: `urlparse`
    # reads `C:\store` as scheme `c`. The shortest real scheme in existence is
    # two characters (`ws`), so a single-character scheme is always a drive
    # letter and never a protocol. A zero-length scheme is a plain relative or
    # absolute path. This is a rule, not an enumeration — every drive letter
    # works without being listed.
    #
    # `ntpath.splitdrive` additionally recognises UNC paths. `ntpath` is used
    # rather than `os.path` so the result does not depend on the host platform:
    # `posixpath.splitdrive` returns no drive for `C:\store`, which would make
    # this branch pass on one platform and not another.
    #
    # Returned directly (ruff SIM103): this is the last branch, so anything that
    # is not a local path is rejected.
    return len(scheme) <= 1 or bool(ntpath.splitdrive(value)[0])


def _redact_credentials(value: str) -> str:
    """Mask any password embedded in a value's userinfo component.

    A rejected value is quoted into an exception message, and an uncaught
    exception's traceback lands wherever stderr was captured — a retained,
    broadly readable job log, say. A value addressing a store can carry a
    password in its userinfo, and such a value is never an encrypted transport,
    so it always takes the reject path. Copying that password into a log would
    give it a new place to live.

    Scheme, host, port, path, and query all survive, because they are what makes
    the message actionable. The username survives too: it is normally a
    non-secret identity, and keeping it distinguishes a wrong user from a wrong
    host. A lone userinfo component with no ``:`` is masked whole, because it
    may itself be a token.

    Only called after `is_secure_endpoint_value` has already parsed this value
    without raising, so parsing here cannot raise either.
    """
    parts = urlsplit(value)
    if "@" not in parts.netloc:
        return value
    userinfo, _, host = parts.netloc.rpartition("@")
    user, separator, _password = userinfo.partition(":")
    masked = f"{user}:***@{host}" if separator else f"***@{host}"
    return urlunsplit(parts._replace(netloc=masked))


def validate_secure_transport(field_name: str, value: str) -> None:
    """Raise if a caller-supplied endpoint value is not acceptable.

    Args:
        field_name: Name of the setting being validated, as the caller knows
            it, so the message identifies what to change.
        value: The endpoint value to validate.

    Raises:
        ValueError: When the value is not acceptable. The message names the
            setting, the offending value with any embedded password masked, and
            the parsed scheme, so it is actionable on its own. There is no
            bypass: every accepted value either encrypts in transit, transmits
            nothing, or transmits only to the machine it runs on.
    """
    if is_secure_endpoint_value(value):
        return

    scheme = urlparse(value).scheme
    allowed = ", ".join(sorted(SECURE_SCHEMES))
    # ValueError, never `sys.exit`: this is a library, and the caller owns the
    # decision of what to do about a bad configuration value.
    raise ValueError(
        f"{field_name} was set to {_redact_credentials(value)!r}, which would "
        f"not transmit over an encrypted transport (parsed scheme: {scheme!r}). "
        f"Use an encrypted endpoint ({allowed}), a local store, or a loopback "
        f"address such as 'http://localhost:5000', which reaches no network."
    )
