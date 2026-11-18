package ai.offgridmobile.localdream

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * GenerationCancelGate: a cancel reaches exactly the generations issued before it.
 * Falsify: reset a shared flag on issue() (the old generationCancelled.set(false)) -> the
 * "queued run stays cancelled" case goes red.
 */
@Suppress("kotlin:S100") // Backtick test names are idiomatic Kotlin
class GenerationCancelGateTest {

    @Test
    fun `a queued run stays cancelled when a newer run is issued after the cancel`() {
        val gate = GenerationCancelGate()
        val a = gate.issue()
        gate.cancelAll()
        val b = gate.issue()
        assertTrue(gate.isCancelled(a))
        assertFalse(gate.isCancelled(b))
    }

    @Test
    fun `cancel covers every run issued so far, queued or running`() {
        val gate = GenerationCancelGate()
        val a = gate.issue()
        val b = gate.issue()
        gate.cancelAll()
        assertTrue(gate.isCancelled(a))
        assertTrue(gate.isCancelled(b))
    }

    @Test
    fun `an idle cancel does not poison the next run`() {
        val gate = GenerationCancelGate()
        gate.cancelAll()
        val a = gate.issue()
        assertFalse(gate.isCancelled(a))
    }

    @Test
    fun `no run is cancelled before any cancel`() {
        val gate = GenerationCancelGate()
        assertFalse(gate.isCancelled(gate.issue()))
    }
}
