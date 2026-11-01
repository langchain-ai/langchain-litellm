;;; Directory Local Variables -*- no-byte-compile: t -*-
;; Runs the ruff pinned in uv.lock through Eglot and formats Python before save.
;; Emacs asks before it applies the `eval' form.
((python-base-mode
  . ((eval . (progn
               (require 'eglot)
               (add-to-list 'eglot-server-programs
                            '(python-base-mode . ("uv" "run" "--frozen" "ruff" "server")))
               (add-hook 'before-save-hook #'eglot-format-buffer nil t)
               (eglot-ensure))))))
