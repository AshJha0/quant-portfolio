package io.quantportfolio.ull.metrics;

import com.sun.management.GarbageCollectionNotificationInfo;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.ArrayList;
import java.util.List;
import javax.management.NotificationEmitter;
import javax.management.NotificationListener;
import javax.management.openmbean.CompositeData;

/**
 * Records every garbage-collection event in the JVM while it is installed:
 * collector name, cause and duration. It listens to the HotSpot GC
 * notifications ({@code com.sun.management.GarbageCollectionNotificationInfo}),
 * which arrive on a JMX thread, so the listener's own allocation never runs
 * on the measured thread.
 *
 * <p>What it measures: the duration the collector reports for each event.
 * For G1 young pauses that is the stop-the-world pause; for concurrent
 * cycles it includes concurrent time, so "pause" in the report means
 * "GC event duration as the JVM reports it". Time-to-safepoint is not
 * included (use {@code -Xlog:safepoint} for that).
 */
public final class GcPauseTracker implements AutoCloseable {
    /** One collector event. */
    public record Event(String collector, String action, String cause, long durationMillis) {}

    private final List<Event> events = new ArrayList<>();
    private final List<NotificationEmitter> emitters = new ArrayList<>();
    private final NotificationListener listener = (n, h) -> {
        if (GarbageCollectionNotificationInfo.GARBAGE_COLLECTION_NOTIFICATION.equals(n.getType())) {
            GarbageCollectionNotificationInfo info =
                GarbageCollectionNotificationInfo.from((CompositeData) n.getUserData());
            synchronized (events) {
                events.add(new Event(info.getGcName(), info.getGcAction(), info.getGcCause(),
                    info.getGcInfo().getDuration()));
            }
        }
    };
    private final long startCount;

    public GcPauseTracker() {
        for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
            if (gc instanceof NotificationEmitter e) {
                e.addNotificationListener(listener, null, null);
                emitters.add(e);
            }
        }
        startCount = collectionCount();
    }

    /** Sum of collection counts over all collectors (polled; catches events before notifications land). */
    public static long collectionCount() {
        long c = 0;
        for (GarbageCollectorMXBean gc : ManagementFactory.getGarbageCollectorMXBeans()) {
            c += Math.max(0, gc.getCollectionCount());
        }
        return c;
    }

    /** Collections since this tracker was created, from the polled counters. */
    public long collectionsSinceStart() {
        return collectionCount() - startCount;
    }

    public List<Event> events() {
        synchronized (events) {
            return List.copyOf(events);
        }
    }

    public long maxPauseMillis() {
        return events().stream().mapToLong(Event::durationMillis).max().orElse(0);
    }

    @Override
    public void close() {
        for (NotificationEmitter e : emitters) {
            try {
                e.removeNotificationListener(listener);
            } catch (javax.management.ListenerNotFoundException ignored) {
                // already removed
            }
        }
    }
}
