package ai.offgridmobile.sdcpp

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * SdCppRunGate: the sd.cpp cancel bookkeeping (SdCppImageModule.generateImage / cancelGeneration).
 * Falsify: let consumeStaleCancel ignore the run's own cancel (the old `cancelRequested = false`
 * before throwing) -> "user cancel before start is never retried" goes red.
 */
@Suppress("kotlin:S100") // Backtick test names are idiomatic Kotlin
class SdCppRunGateTest {

    private fun sendCounter(): Pair<IntArray, () -> Boolean> {
        val sent = intArrayOf(0)
        return sent to { sent[0]++; true }
    }

    @Test
    fun `user cancel before start is never retried even after a stale native cancel`() {
        val gate = SdCppRunGate()
        // Run 1 is inside native, the user cancels, run 1 still returns normally -> stale flag set.
        val r1 = gate.issue()
        assertTrue(gate.enterNative(r1))
        val (sent, send) = sendCounter()
        gate.cancelAll(send)
        assertEquals(1, sent[0])
        gate.exitNative()
        gate.nativeReturned()
        gate.finish()
        // Run 2 is queued, then the user cancels it before it starts.
        val r2 = gate.issue()
        gate.cancelAll(send)
        assertEquals("no native cancel while nothing is in native", 1, sent[0])
        assertFalse(gate.enterNative(r2))
        assertFalse("a user-cancelled run must not take the stale-cancel retry", gate.consumeStaleCancel(r2))
        gate.finish()
    }

    @Test
    fun `stale native cancel from the previous run allows exactly one retry`() {
        val gate = SdCppRunGate()
        val r1 = gate.issue()
        assertTrue(gate.enterNative(r1))
        gate.cancelAll { true }
        gate.exitNative()
        gate.nativeReturned()
        gate.finish()
        val r2 = gate.issue()
        assertTrue(gate.enterNative(r2))
        gate.exitNative()
        assertTrue(gate.consumeStaleCancel(r2))
        assertFalse(gate.consumeStaleCancel(r2))
        assertTrue(gate.enterNative(r2))
    }

    @Test
    fun `a cancel while two runs are queued cancels both, a later run is clean`() {
        val gate = SdCppRunGate()
        val r1 = gate.issue()
        val r2 = gate.issue()
        gate.finish() // run 1 finishes first: its finally must not drop the cancel for run 2
        gate.cancelAll { true }
        assertTrue(gate.isCancelled(r1))
        assertTrue(gate.isCancelled(r2))
        assertFalse(gate.enterNative(r2))
        gate.finish()
        val r3 = gate.issue()
        assertTrue(gate.enterNative(r3))
    }

    @Test
    fun `a newer generateImage does not clear a cancel aimed at a queued run`() {
        val gate = SdCppRunGate()
        val r1 = gate.issue()
        gate.cancelAll { true }
        val r2 = gate.issue()
        assertFalse(gate.enterNative(r1))
        assertTrue(gate.enterNative(r2))
    }

    @Test
    fun `an idle cancel is a no-op`() {
        val gate = SdCppRunGate()
        var called = false
        gate.cancelAll { called = true; true }
        assertFalse(called)
        val r1 = gate.issue()
        assertTrue(gate.enterNative(r1))
    }

    @Test
    fun `native cancel is sent only while a run is inside native`() {
        val gate = SdCppRunGate()
        val r1 = gate.issue()
        var sent = 0
        gate.cancelAll { sent++; true }
        assertEquals(0, sent)
        assertFalse(gate.enterNative(r1))
        val r2 = gate.issue()
        assertTrue(gate.enterNative(r2))
        gate.cancelAll { sent++; true }
        assertEquals(1, sent)
    }
}
