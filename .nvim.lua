-- Project config, loaded when 'exrc' is on; Neovim asks you to trust it first.
-- Runs the ruff pinned in uv.lock as the language server and formats on save.
if vim.fn.has("nvim-0.11") == 0 then
  return
end

vim.lsp.config("ruff", {
  cmd = { "uv", "run", "--frozen", "ruff", "server" },
  filetypes = { "python" },
  root_markers = { "pyproject.toml", ".git" },
})
vim.lsp.enable("ruff")

vim.api.nvim_create_autocmd("BufWritePre", {
  group = vim.api.nvim_create_augroup("langchain_litellm_ruff", { clear = true }),
  pattern = { "*.py", "*.pyi" },
  callback = function(args)
    vim.lsp.buf.format({ bufnr = args.buf, name = "ruff", async = false })
  end,
})
