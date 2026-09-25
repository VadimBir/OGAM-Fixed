import {
  getToolsAsOpenAISchema,
  buildToolSystemPromptHint,
  buildCustomSkillPromptHint,
  buildToolBlockPromptHint,
  setSkillOverrides,
  clampRepeat,
} from '../../../../src/services/tools/registry';

describe('skill-card overrides (editable tool text the model receives)', () => {
  afterEach(() => setSkillOverrides({}, [])); // reset global registry state between tests

  it('an edited description replaces the model-facing tool description in the schema', () => {
    const before = getToolsAsOpenAISchema(['web_search'])[0].function.description;
    setSkillOverrides({ web_search: { description: 'MY CUSTOM SEARCH INSTRUCTIONS' } }, []);
    const after = getToolsAsOpenAISchema(['web_search'])[0].function.description;

    expect(after).toBe('MY CUSTOM SEARCH INSTRUCTIONS');
    expect(after).not.toBe(before);
  });

  it('an edited description replaces the text in the system-prompt hint', () => {
    setSkillOverrides({ web_search: { description: 'EDITED HINT TEXT' } }, []);
    const hint = buildToolSystemPromptHint(['web_search']);
    expect(hint).toContain('web_search: EDITED HINT TEXT');
  });

  it('a blank/whitespace override falls back to the built-in default', () => {
    const def = getToolsAsOpenAISchema(['web_search'])[0].function.description;
    setSkillOverrides({ web_search: { description: '   ' } }, []);
    expect(getToolsAsOpenAISchema(['web_search'])[0].function.description).toBe(def);
  });

  it('enabled custom skills appear in the hint; disabled ones do not', () => {
    setSkillOverrides({}, [
      { id: 's1', name: 'Roleplay', description: 'always narrate in third person', enabled: true },
      { id: 's2', name: 'Secret', description: 'should not appear', enabled: false },
    ]);
    const hint = buildCustomSkillPromptHint();
    expect(hint).toContain('<skill name="Roleplay">\nalways narrate in third person\n</skill>');
    expect(hint).not.toContain('should not appear');
    // Custom skills are prompt-only: never in the tool-list hint.
    expect(buildToolSystemPromptHint(['web_search'])).not.toContain('Roleplay');
  });

  it('custom skills are framed as standing SYSTEM instructions inside <skills>', () => {
    setSkillOverrides({}, [{ id: 's1', name: 'A "q"', description: 'do x', enabled: true }]);
    const hint = buildCustomSkillPromptHint();
    expect(hint.startsWith('\n\n<skills>\n')).toBe(true);
    expect(hint).toContain('standing SYSTEM instructions');
    expect(hint).toContain('<skill name="A \'q\'">');
    expect(hint.endsWith('</skills>')).toBe(true);
  });

  it('no enabled custom skill → empty hint', () => {
    setSkillOverrides({}, [{ id: 's1', name: 'x', description: 'y', enabled: false }]);
    expect(buildCustomSkillPromptHint()).toBe('');
  });

  it("multiplier 'content' repeats the text inside ONE <skill> block", () => {
    setSkillOverrides({}, [{ id: 's1', name: 'R', description: 'obey', enabled: true, repeat: 3 }]);
    const hint = buildCustomSkillPromptHint();
    expect(hint.match(/<skill name=/g)).toHaveLength(1);
    expect(hint).toContain('<skill name="R">\nobey\nobey\nobey\n</skill>');
  });

  it("multiplier 'block' repeats the whole <skill> block", () => {
    setSkillOverrides({}, [
      { id: 's1', name: 'R', description: 'obey', enabled: true, repeat: 2, repeatMode: 'block' },
    ]);
    const hint = buildCustomSkillPromptHint();
    expect(hint.match(/<skill name="R">\nobey\n<\/skill>/g)).toHaveLength(2);
    expect(hint.match(/<skills>/g)).toHaveLength(1);
  });

  it("tool 'content': description repeated inside the schema + text list; no system blocks", () => {
    setSkillOverrides({ web_search: { description: 'S', repeat: 2 } }, []);
    const schema = getToolsAsOpenAISchema(['web_search']);
    expect(schema).toHaveLength(1);
    expect(schema[0].function.description).toBe('S\nS');
    expect(buildToolSystemPromptHint(['web_search'])).toContain('- web_search: S\nS\n');
    expect(buildToolBlockPromptHint(['web_search'])).toBe('');
  });

  it("tool 'block': schema/text list once + N standalone <tool> system blocks (every engine)", () => {
    setSkillOverrides({ web_search: { description: 'S', repeat: 2, repeatMode: 'block' } }, []);
    const schema = getToolsAsOpenAISchema(['web_search']);
    expect(schema).toHaveLength(1); // never a duplicate function
    expect(schema[0].function.description).toBe('S');
    expect(buildToolSystemPromptHint(['web_search'])).toContain('- web_search: S\n  arguments:');
    const hint = buildToolBlockPromptHint(['web_search']);
    expect(hint).toContain('<tool_instructions>');
    expect(hint.match(/<tool name="web_search">\nS\n<\/tool>/g)).toHaveLength(2);
  });

  it('block hint only for ENABLED tools in block mode', () => {
    setSkillOverrides({ web_search: { description: 'S', repeat: 3, repeatMode: 'block' } }, []);
    expect(buildToolBlockPromptHint([])).toBe('');
  });

  it('clampRepeat bounds the multiplier to 1..5', () => {
    expect([clampRepeat(undefined), clampRepeat(0), clampRepeat(2.4), clampRepeat(99), clampRepeat(NaN)])
      .toEqual([1, 1, 2, 5, 1]);
  });

  it('custom skills never leak into the callable tool schema', () => {
    setSkillOverrides({}, [
      { id: 's1', name: 'Roleplay', description: 'x', enabled: true },
    ]);
    const names = getToolsAsOpenAISchema(['web_search']).map(t => t.function.name);
    expect(names).toEqual(['web_search']);
  });

  it('text hint carries the call template, argument names and a literal example', () => {
    const hint = buildToolSystemPromptHint(['generate_image']);
    expect(hint).toContain('<tool_call>{"name": "TOOL_NAME", "arguments": {"ARG": "VALUE"}}</tool_call>');
    expect(hint).toContain('prompt (string, required)');
    expect(hint).toContain('example: <tool_call>{"name":"generate_image","arguments":{"prompt":"..."}}</tool_call>');
  });
});
