//! W30 book acceptance: the fast book equals the reference after every
//! message of generated days, including mid-day snapshot recovery.
//! Messages are decoded from QPBF bytes, so the decoder is in the loop.
//! Set `BOOK_DAY_N` to change the per-seed message count (default 100,000).

use qp_ull_book::Book;
use qp_ull_book::reference::RefBook;
use qp_ull_qpbf::generator::{GenConfig, Generator};
use qp_ull_qpbf::{Msg, Packet, PacketWriter, Side};

fn check_equal(b: &Book, r: &RefBook, at: u64) {
    for side in [Side::Buy, Side::Sell] {
        let mut fast = Vec::with_capacity(b.len());
        b.for_each_order(side, |p, id, q| fast.push((p, id, q)));
        assert_eq!(
            fast,
            r.orders(side),
            "side {side:?} differs at message {at}"
        );
    }
    assert_eq!(b.best_bid(), r.best(Side::Buy), "best bid at {at}");
    assert_eq!(b.best_ask(), r.best(Side::Sell), "best ask at {at}");
}

fn run_day(seed: u64, n: u64, snapshot_every: u64) {
    let cfg = GenConfig {
        seed,
        instruments: 3,
        max_live: 250,
        ..GenConfig::default()
    };
    let n_inst = cfg.instruments as usize;
    let mid = cfg.start_mid;
    let mut g = Generator::new(cfg);
    let mut fast: Vec<Book> = (0..n_inst)
        .map(|_| Book::new(mid - 30_000, 60_000, 4_096))
        .collect();
    let mut refs: Vec<RefBook> = vec![RefBook::default(); n_inst];
    let mut w = PacketWriter::new(1, 1);
    let mut pending: Vec<Msg> = Vec::new();
    let mut applied = 0u64;
    let mut before_snap: Option<RefBook> = None;
    let mut in_snap = false;
    let mut snaps = 0u32;
    let mut drain = |pkt: &[u8], fast: &mut Vec<Book>, refs: &mut Vec<RefBook>| {
        for (_, m) in Packet::parse(pkt).unwrap().messages() {
            if m.instrument() == 0 {
                // Session-level heartbeat: no book.
                assert!(matches!(m, Msg::Heartbeat { .. }));
                applied += 1;
                continue;
            }
            let i = (m.instrument() - 1) as usize;
            if matches!(m, Msg::Snapshot { .. }) && !in_snap {
                // The wire has no FIRST flag: the receiver clears on the
                // first snapshot message it applies.
                before_snap = Some(refs[i].clone());
                fast[i].clear();
                refs[i].clear();
                in_snap = true;
            }
            let a = fast[i].apply(&m);
            let b = refs[i].apply(&m);
            assert_eq!(a, b, "apply result differs on {m:?}");
            assert_eq!(a, Ok(()), "generator produced an invalid message {m:?}");
            applied += 1;
            check_equal(&fast[i], &refs[i], applied);
            if let Msg::Snapshot { flags, .. } = m
                && flags & qp_ull_qpbf::SNAP_LAST != 0
            {
                // A snapshot of the live state must rebuild the same book,
                // priority order included.
                let pre = before_snap.take().expect("snapshot LAST without start");
                in_snap = false;
                assert_eq!(pre, refs[i], "snapshot changed the book at {applied}");
                snaps += 1;
            }
        }
    };
    for k in 1..=n {
        pending.push(g.next_msg());
        if snapshot_every > 0 && k % snapshot_every == 0 {
            let inst = 1 + (k / snapshot_every % n_inst as u64) as u32;
            let seq = w.next_seq() + pending.len() as u64 - 1;
            pending.extend(g.snapshot(inst, seq));
        }
        for m in pending.drain(..) {
            if w.would_overflow(&m) {
                let mut sink = |p: &[u8]| drain(p, &mut fast, &mut refs);
                w.flush(&mut sink);
            }
            w.push(&m);
        }
    }
    let mut sink = |p: &[u8]| drain(p, &mut fast, &mut refs);
    w.flush(&mut sink);
    drop(drain);
    if let Some(expected) = n.checked_div(snapshot_every) {
        assert_eq!(snaps as u64, expected);
    }
    let ledger_total: u64 = g.ledger().iter().sum();
    assert_eq!(applied, ledger_total);
    for (i, b) in fast.iter().enumerate() {
        assert_eq!(b.len(), g.live_orders(i as u32 + 1));
    }
}

fn n_from_env() -> u64 {
    std::env::var("BOOK_DAY_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(100_000)
}

#[test]
fn generated_day_seed_1() {
    run_day(1, n_from_env(), 0);
}

#[test]
fn generated_day_seed_2_with_snapshots() {
    run_day(2, n_from_env(), 25_000);
}

#[test]
fn generated_day_seed_3_with_frequent_snapshots() {
    run_day(3, n_from_env() / 3, 997);
}

#[test]
fn empty_book_snapshot_clears() {
    let mut b = Book::new(0, 100, 8);
    b.add(5, Side::Sell, 50, 3).unwrap();
    b.apply(&Msg::Snapshot {
        instrument: 1,
        ts_ns: 0,
        flags: qp_ull_qpbf::SNAP_LAST,
        order_id: 0,
        side: Side::Buy,
        price: 0,
        qty: 0,
        as_of_seq: 9,
    })
    .unwrap();
    assert_eq!(b.len(), 1, "end marker alone adds and removes nothing");
    b.clear();
    assert!(b.is_empty());
    assert_eq!(b.best_ask(), None);
    b.add(5, Side::Sell, 50, 3).unwrap();
}
