//! Raw recorder and replay.
//!
//! Plain idea: venues change their message formats, so the recorder keeps
//! the exact bytes received, with the receive time and the line, and parsing
//! happens on replay. A recorded day can then be re-run through any later
//! version of the parser or book.
//!
//! Formal statement (file format `QPRR1`, all integers little-endian):
//! the 6-byte magic `b"QPRR1\n"`, then frames of
//! `recv_ns: u64, line: u8, kind: u8, len: u32, bytes[len]`.
//! `kind` 0 is a stream message, 1 a REST snapshot reply. A truncated last
//! frame (a crash mid-write) ends replay with `Truncated`, never a panic; a
//! frame longer than `MAX_FRAME` is rejected as corrupt.
//!
//! Worked example: one frame at 1 ns on line B, kind 0, payload `{}` is
//! `01 00 00 00 00 00 00 00 | 01 | 00 | 02 00 00 00 | 7b 7d` after the magic.

use std::io::{self, Read, Write};

/// File magic.
pub const MAGIC: &[u8; 6] = b"QPRR1\n";
/// Largest accepted frame payload (16 MiB; a Binance 5000-level snapshot is ~300 KB).
pub const MAX_FRAME: u32 = 16 << 20;

/// Stream message.
pub const KIND_STREAM: u8 = 0;
/// REST snapshot reply.
pub const KIND_SNAPSHOT: u8 = 1;

/// One recorded frame.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Receive time, ns since the Unix epoch (or any monotonic origin).
    pub recv_ns: u64,
    /// Line: 0 = A, 1 = B.
    pub line: u8,
    /// [`KIND_STREAM`] or [`KIND_SNAPSHOT`].
    pub kind: u8,
    /// Raw bytes as received.
    pub bytes: Vec<u8>,
}

/// Appends frames to a writer.
pub struct Recorder<W: Write> {
    w: W,
}

impl<W: Write> Recorder<W> {
    /// Write the magic and return a recorder.
    pub fn new(mut w: W) -> io::Result<Self> {
        w.write_all(MAGIC)?;
        Ok(Self { w })
    }
    /// Append one frame.
    pub fn record(&mut self, recv_ns: u64, line: u8, kind: u8, bytes: &[u8]) -> io::Result<()> {
        let len = u32::try_from(bytes.len()).ok().filter(|&n| n <= MAX_FRAME);
        let len =
            len.ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "frame too large"))?;
        let mut h = [0u8; 14];
        h[..8].copy_from_slice(&recv_ns.to_le_bytes());
        h[8] = line;
        h[9] = kind;
        h[10..].copy_from_slice(&len.to_le_bytes());
        self.w.write_all(&h)?;
        self.w.write_all(bytes)
    }
    /// Flush and return the writer.
    pub fn finish(mut self) -> io::Result<W> {
        self.w.flush()?;
        Ok(self.w)
    }
}

/// Replay errors.
#[derive(Debug)]
pub enum ReplayError {
    /// Missing or wrong magic.
    BadMagic,
    /// The file ends inside a frame.
    Truncated,
    /// Frame length over [`MAX_FRAME`].
    Corrupt,
    /// Underlying I/O error.
    Io(io::Error),
}

/// Reads frames back in order.
pub struct Replay<R: Read> {
    r: R,
}

fn read_full<R: Read>(r: &mut R, buf: &mut [u8]) -> Result<bool, ReplayError> {
    // Ok(false) on clean EOF before the first byte, Truncated on partial.
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => {
                return if n == 0 {
                    Ok(false)
                } else {
                    Err(ReplayError::Truncated)
                };
            }
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(ReplayError::Io(e)),
        }
    }
    Ok(true)
}

impl<R: Read> Replay<R> {
    /// Check the magic.
    pub fn new(mut r: R) -> Result<Self, ReplayError> {
        let mut m = [0u8; 6];
        if !read_full(&mut r, &mut m)? || &m != MAGIC {
            return Err(ReplayError::BadMagic);
        }
        Ok(Self { r })
    }
}

impl<R: Read> Iterator for Replay<R> {
    type Item = Result<Frame, ReplayError>;
    fn next(&mut self) -> Option<Self::Item> {
        let mut h = [0u8; 14];
        match read_full(&mut self.r, &mut h) {
            Ok(false) => return None,
            Err(e) => return Some(Err(e)),
            Ok(true) => {}
        }
        let len = u32::from_le_bytes([h[10], h[11], h[12], h[13]]);
        if len > MAX_FRAME {
            return Some(Err(ReplayError::Corrupt));
        }
        let mut bytes = vec![0u8; len as usize];
        match read_full(&mut self.r, &mut bytes) {
            Ok(true) => {}
            Ok(false) if len == 0 => {}
            Ok(false) | Err(ReplayError::Truncated) => return Some(Err(ReplayError::Truncated)),
            Err(e) => return Some(Err(e)),
        }
        let mut t = [0u8; 8];
        t.copy_from_slice(&h[..8]);
        Some(Ok(Frame {
            recv_ns: u64::from_le_bytes(t),
            line: h[8],
            kind: h[9],
            bytes,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worked_example_bytes() {
        let mut r = Recorder::new(Vec::new()).unwrap();
        r.record(1, 1, KIND_STREAM, b"{}").unwrap();
        let v = r.finish().unwrap();
        assert_eq!(
            &v[6..],
            &[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, b'{', b'}']
        );
        let f: Vec<_> = Replay::new(&v[..]).unwrap().map(Result::unwrap).collect();
        assert_eq!(
            f,
            vec![Frame {
                recv_ns: 1,
                line: 1,
                kind: 0,
                bytes: b"{}".to_vec()
            }]
        );
    }

    #[test]
    fn truncation_and_corruption_do_not_panic() {
        let mut r = Recorder::new(Vec::new()).unwrap();
        r.record(5, 0, KIND_SNAPSHOT, b"hello").unwrap();
        let v = r.finish().unwrap();
        for cut in 7..v.len() {
            let got: Vec<_> = Replay::new(&v[..cut]).unwrap().collect();
            assert!(
                matches!(got.as_slice(), [Err(ReplayError::Truncated)]),
                "cut {cut}"
            );
        }
        let mut bad = v.clone();
        bad[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(matches!(
            Replay::new(&bad[..]).unwrap().next(),
            Some(Err(ReplayError::Corrupt))
        ));
        assert!(matches!(
            Replay::new(&b"nope!!"[..]),
            Err(ReplayError::BadMagic)
        ));
    }
}
