-- Memory MCP Browser for Neovim
-- Telescope picker for browsing memories from memory-mcp server
-- Keymap: <leader>sm to open memory browser

local M = {}

local function browse_memories()
  local ok_telescope, _ = pcall(require, "telescope")
  if not ok_telescope then
    vim.notify("Telescope not found", vim.log.levels.ERROR)
    return
  end

  local pickers = require("telescope.pickers")
  local finders = require("telescope.finders")
  local conf = require("telescope.config").values
  local actions = require("telescope.actions")
  local action_state = require("telescope.actions.state")
  local previewers = require("telescope.previewers")

  -- Get memories via Python
  local handle = io.popen([[python3 -c "
import json
from memory_mcp.storage import connect, list_memories
conn = connect()
mems = list_memories(conn, None, None, None, 0, None, None, None, None, None)
conn.close()
print(json.dumps(mems))
" 2>/dev/null]])

  if not handle then
    vim.notify("Failed to run memory query", vim.log.levels.ERROR)
    return
  end

  local result = handle:read("*a")
  handle:close()

  local ok, memories = pcall(vim.json.decode, result)
  if not ok or not memories then
    vim.notify("Failed to parse memories: " .. (result or "empty"), vim.log.levels.ERROR)
    return
  end

  pickers.new({}, {
    prompt_title = "Memory Browser",
    finder = finders.new_table({
      results = memories,
      entry_maker = function(m)
        local tags = table.concat(m.tags or {}, ", ")
        local preview = m.content:sub(1, 50):gsub("\n", " ")
        return {
          value = m,
          display = string.format("#%-3d │ %-20s │ %s", m.id, tags:sub(1, 20), preview),
          ordinal = m.content .. " " .. tags,
        }
      end,
    }),
    sorter = conf.generic_sorter({}),
    previewer = previewers.new_buffer_previewer({
      title = "Memory Content",
      define_preview = function(self, entry)
        local m = entry.value
        local lines = {
          "# Memory #" .. m.id,
          "",
          "**Tags:** " .. table.concat(m.tags or {}, ", "),
          "**Created:** " .. (m.created_at or ""),
          "",
          "---",
          "",
        }
        for line in m.content:gmatch("[^\n]*") do
          table.insert(lines, line)
        end
        vim.api.nvim_buf_set_lines(self.state.bufnr, 0, -1, false, lines)
        vim.bo[self.state.bufnr].filetype = "markdown"
      end,
    }),
    attach_mappings = function(prompt_bufnr)
      actions.select_default:replace(function()
        local sel = action_state.get_selected_entry()
        actions.close(prompt_bufnr)
        vim.cmd("enew")
        vim.api.nvim_buf_set_lines(0, 0, -1, false, vim.split(sel.value.content, "\n"))
        vim.bo.filetype = "markdown"
        vim.bo.buftype = "nofile"
        vim.bo.bufhidden = "wipe"
      end)
      return true
    end,
  }):find()
end

-- Plugin spec for lazy.nvim / kickstart.nvim
return {
  "nvim-telescope/telescope.nvim",
  dependencies = { "nvim-lua/plenary.nvim" },
  keys = {
    { "<leader>sm", browse_memories, desc = "[S]earch [M]emories" },
  },
  config = function()
    M.browse = browse_memories
  end,
}
