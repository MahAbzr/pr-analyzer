import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import CodeAnalyzer from '../App';

globalThis.fetch = vi.fn();

describe('CodeAnalyzer', () => {
  beforeEach(() => {
    fetch.mockClear();
    // Mock the initial fetchAnalyses call that happens in useEffect
    fetch.mockResolvedValue({
      ok: true,
      json: async () => []
    });
  });

  it('renders the component', async () => {
    render(<CodeAnalyzer />);

    // Wait for the component to finish loading
    await waitFor(() => {
      expect(screen.getByText(/Code Security Analyzer/i)).toBeInTheDocument();
    });
  });

  it('handles code input', async () => {
    render(<CodeAnalyzer />);

    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Paste your code/i)).toBeInTheDocument();
    });

    const textarea = screen.getByPlaceholderText(/Paste your code/i);
    fireEvent.change(textarea, { target: { value: 'def test():\n    pass' } });
    expect(textarea.value).toBe('def test():\n    pass');
  });

  it('submits code for analysis', async () => {
    render(<CodeAnalyzer />);

    // Wait for initial render
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Paste your code/i)).toBeInTheDocument();
    });

    // Mock the analyze API call (second fetch call)
    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        id: 'test-123',
        risk_score_before: 0.5,
        risk_score_after: 0.2,
        fixed_code: 'def test():\n    pass'
      })
    });

    const textarea = screen.getByPlaceholderText(/Paste your code/i);
    const button = screen.getByRole('button', { name: /Analyze Code/i });

    fireEvent.change(textarea, { target: { value: 'def test(): pass' } });
    fireEvent.click(button);

    // Wait for the analyze call (should be called after the initial fetchAnalyses)
    await waitFor(() => {
      expect(fetch).toHaveBeenCalledTimes(2);
    });
  });
});
