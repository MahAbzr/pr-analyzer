import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import CodeAnalyzer from '../App';

globalThis.fetch = vi.fn();

describe('CodeAnalyzer', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  it('renders the component', () => {
    render(<CodeAnalyzer />);
    expect(screen.getByText(/Code Risk Analyzer/i)).toBeInTheDocument();
  });

  it('handles code input', () => {
    render(<CodeAnalyzer />);
    const textarea = screen.getByPlaceholderText(/Paste your code/i);

    fireEvent.change(textarea, { target: { value: 'def test():\n    pass' } });
    expect(textarea.value).toBe('def test():\n    pass');
  });

  it('submits code for analysis', async () => {
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        risk_score_before: 0.5,
        risk_score_after: 0.2,
        fixed_code: 'def test():\n    pass'
      })
    });

    render(<CodeAnalyzer />);
    const textarea = screen.getByPlaceholderText(/Paste your code/i);
    const button = screen.getByRole('button', { name: /Analyze/i });

    fireEvent.change(textarea, { target: { value: 'def test(): pass' } });
    fireEvent.click(button);

    await waitFor(() => {
      expect(fetch).toHaveBeenCalledTimes(1);
    });
  });
});
