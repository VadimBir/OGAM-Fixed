import {
  getToolsAsOpenAISchema,
  buildToolSystemPromptHint,
  setSkillOverrides,
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
    const hint = buildToolSystemPromptHint(['web_search']);
    expect(hint).toContain('Roleplay: always narrate in third person');
    expect(hint).not.toContain('should not appear');
  });

  it('custom skills never leak into the callable tool schema', () => {
    setSkillOverrides({}, [
      { id: 's1', name: 'Roleplay', description: 'x', enabled: true },
    ]);
    const names = getToolsAsOpenAISchema(['web_search']).map(t => t.function.name);
    expect(names).toEqual(['web_search']);
  });
});
