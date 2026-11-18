/**
 * SliderSetting Component Tests
 *
 * The shared numeric control used across every generation/model/TTS settings
 * screen: a draggable slider (coarse, step-snapped) with a live value you can
 * tap to type an exact number (precise). Covers display formatting, tap-to-edit
 * with clamping/validation, and slider commit-on-release.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { SliderSetting } from '../../../src/components/SliderSetting';

describe('SliderSetting', () => {
  const baseProps = {
    testID: 'temp',
    label: 'Temperature',
    value: 0.7,
    min: 0,
    max: 2,
    step: 0.05,
    onChange: jest.fn(),
  };

  beforeEach(() => jest.clearAllMocks());

  it('renders label, description and warning', () => {
    const { getByText } = render(
      <SliderSetting {...baseProps} description="Higher = more creative" warning="Uses lots of RAM" />,
    );
    expect(getByText('Temperature')).toBeTruthy();
    expect(getByText('Higher = more creative')).toBeTruthy();
    expect(getByText('Uses lots of RAM')).toBeTruthy();
  });

  it('does not render description/warning when not provided', () => {
    const { queryByText } = render(<SliderSetting {...baseProps} />);
    expect(queryByText('Higher = more creative')).toBeNull();
  });

  it('formats the displayed value with formatValue', () => {
    const { getByTestId } = render(
      <SliderSetting {...baseProps} formatValue={(v) => `${v.toFixed(2)}x`} />,
    );
    expect(getByTestId('temp-value').props.children).toBe('0.70x');
  });

  it('defaults to 2 decimals when step < 1', () => {
    const { getByTestId } = render(<SliderSetting {...baseProps} />);
    expect(getByTestId('temp-value').props.children).toBe('0.70');
  });

  it('defaults to 0 decimals when step >= 1', () => {
    const { getByTestId } = render(
      <SliderSetting testID="tok" label="Max Tokens" value={1024} min={64} max={8192} step={64} onChange={jest.fn()} />,
    );
    expect(getByTestId('tok-value').props.children).toBe('1024');
  });

  // ─── Tap-to-edit ──────────────────────────────────────────────────────────

  it('reveals a text input pre-filled with the raw value when the value is tapped', () => {
    const { getByTestId } = render(
      <SliderSetting {...baseProps} formatValue={(v) => `${v.toFixed(2)}x`} />,
    );
    fireEvent.press(getByTestId('temp-value-button'));
    // The editable field shows the raw number, not the formatted display.
    expect(getByTestId('temp-input').props.value).toBe('0.7');
  });

  it('commits an exact typed value on submit', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(<SliderSetting {...baseProps} onChange={onChange} />);
    fireEvent.press(getByTestId('temp-value-button'));
    fireEvent.changeText(getByTestId('temp-input'), '0.73');
    fireEvent(getByTestId('temp-input'), 'submitEditing');
    // Typed values keep full precision (not snapped to the 0.05 step grid).
    expect(onChange).toHaveBeenCalledWith(0.73);
  });

  it('clamps a typed value above max', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(<SliderSetting {...baseProps} onChange={onChange} />);
    fireEvent.press(getByTestId('temp-value-button'));
    fireEvent.changeText(getByTestId('temp-input'), '9');
    fireEvent(getByTestId('temp-input'), 'submitEditing');
    expect(onChange).toHaveBeenCalledWith(2);
  });

  it('clamps a typed value below min', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(<SliderSetting {...baseProps} onChange={onChange} />);
    fireEvent.press(getByTestId('temp-value-button'));
    fireEvent.changeText(getByTestId('temp-input'), '-3');
    fireEvent(getByTestId('temp-input'), 'submitEditing');
    expect(onChange).toHaveBeenCalledWith(0);
  });

  it('ignores invalid (non-numeric) input', () => {
    const onChange = jest.fn();
    const { getByTestId, queryByTestId } = render(<SliderSetting {...baseProps} onChange={onChange} />);
    fireEvent.press(getByTestId('temp-value-button'));
    fireEvent.changeText(getByTestId('temp-input'), 'abc');
    fireEvent(getByTestId('temp-input'), 'submitEditing');
    expect(onChange).not.toHaveBeenCalled();
    // Editing closes regardless, returning to the tappable value.
    expect(queryByTestId('temp-value')).toBeTruthy();
  });

  // ─── Slider drag ────────────────────────────────────────────────────────────

  it('commits a clamped, snapped value on slider release', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(<SliderSetting {...baseProps} onChange={onChange} />);
    fireEvent(getByTestId('temp-slider'), 'slidingComplete', 5);
    expect(onChange).toHaveBeenCalledWith(2);
  });

  it('snaps an off-grid slider value to the nearest step on release', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(<SliderSetting {...baseProps} onChange={onChange} />);
    fireEvent(getByTestId('temp-slider'), 'slidingComplete', 0.72);
    expect(onChange).toHaveBeenCalledWith(0.7);
  });
});

describe('nativeSliderStep (slider 5.1.2 allocates one entry per step on every render)', () => {
  const { nativeSliderStep, MAX_SLIDER_POSITIONS } = require('../../../src/components/SliderSetting');
  it('keeps the real step while the range has <= 1000 positions', () => {
    expect(nativeSliderStep(0, 2, 0.05)).toBe(0.05);
    expect(nativeSliderStep(0, 1000, 1)).toBe(1);
  });
  it('coarsens a huge range to a multiple of the step with <= 1000 positions', () => {
    const s = nativeSliderStep(-1, 2147483647, 1);
    expect(Number.isInteger(s)).toBe(true);
    expect((2147483647 + 1) / s).toBeLessThanOrEqual(MAX_SLIDER_POSITIONS);
    expect(nativeSliderStep(0, 4096, 1)).toBe(5);
  });
  it('passes through a zero step or an empty range', () => {
    expect(nativeSliderStep(0, 10, 0)).toBe(0);
    expect(nativeSliderStep(5, 5, 1)).toBe(1);
  });
  it('the seed slider still commits an exact integer and typing stays exact', () => {
    const onChange = jest.fn();
    const { getByTestId } = render(
      <SliderSetting testID="seed" label="Seed" value={-1} min={-1} max={2147483647} step={1} onChange={onChange} />,
    );
    expect(getByTestId('seed-slider').props.step).toBe(nativeSliderStep(-1, 2147483647, 1));
    fireEvent(getByTestId('seed-slider'), 'slidingComplete', 12345.4);
    expect(onChange).toHaveBeenLastCalledWith(12345);
    fireEvent.press(getByTestId('seed-value-button'));
    fireEvent.changeText(getByTestId('seed-input'), '987654321');
    fireEvent(getByTestId('seed-input'), 'submitEditing');
    expect(onChange).toHaveBeenLastCalledWith(987654321);
  });
});
