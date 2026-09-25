; fix_microstation_blocks.lsp
;
; Fixes duplicate block names from MicroStation DWG export.
; MicroStation creates BaseName_1, BaseName_2 ... for each unique scale.
; This routine remaps every INSERT to the base block and adjusts its
; X/Y/Z scale so the visual size is unchanged.
;
; Usage:
;   (load "fix_microstation_blocks.lsp")
;   FIXDUPBLOCKS
;
; Afterwards: PURGE -> All to remove the unused _N definitions.


; Return 1-based position of last underscore in S, or 0 if none.
(defun fdb-last-us (s / i pos)
  (setq pos 0 i 1)
  (while (<= i (strlen s))
    (if (= (substr s i 1) "_") (setq pos i))
    (setq i (1+ i))
  )
  pos
)


; Return T if S is non-empty and every character is a digit 0-9.
(defun fdb-digits-p (s / i c ok)
  (if (= (strlen s) 0)
    nil
    (progn
      (setq ok T i 1)
      (while (and ok (<= i (strlen s)))
        (setq c (ascii (substr s i 1)))
        (if (or (< c 48) (> c 57)) (setq ok nil))
        (setq i (1+ i))
      )
      ok
    )
  )
)


; If NAME ends with _<digits> return the part before the suffix, else nil.
(defun fdb-base-name (name / upos suffix base)
  (setq upos (fdb-last-us name))
  (if (< upos 2)
    nil
    (progn
      (setq suffix (substr name (1+ upos))
            base   (substr name 1 (1- upos)))
      (if (and (fdb-digits-p suffix) (> (strlen base) 0))
        base
        nil
      )
    )
  )
)


; Replace or insert a DXF group code in an entity data list.
(defun fdb-setdxf (ed code val)
  (if (assoc code ed)
    (subst (cons code val) (assoc code ed) ed)
    (append ed (list (cons code val)))
  )
)


; Insert BLKNAME at origin with scale 1 and return bounding box
; as list ((x1 y1 z1) (x2 y2 z2)), or nil on any error.
(defun fdb-bbox (blkname / adoc msp ref lo hi res)
  (setq res nil)
  (vl-catch-all-apply
    (function
      (lambda ()
        (setq adoc (vla-get-activedocument (vlax-get-acad-object))
              msp  (vla-get-modelspace adoc)
              ref  (vla-insertblock msp
                     (vlax-3d-point 0.0 0.0 0.0)
                     blkname 1.0 1.0 1.0 0.0))
        (vla-getboundingbox ref 'lo 'hi)
        (vla-delete ref)
        (setq res (list (vlax-safearray->list lo)
                        (vlax-safearray->list hi)))
      )
    )
    nil
  )
  res
)


; Return scale ratio: size(NUM-NAME) / size(BASE-NAME).
; Falls back to 1.0 when bounding box cannot be computed.
(defun fdb-ratio (base-name num-name / be ne bw bh nw nh)
  (setq be (fdb-bbox base-name)
        ne (fdb-bbox num-name))
  (if (and be ne)
    (progn
      (setq bw (abs (- (car  (cadr be)) (car  (car be))))
            bh (abs (- (cadr (cadr be)) (cadr (car be))))
            nw (abs (- (car  (cadr ne)) (car  (car ne))))
            nh (abs (- (cadr (cadr ne)) (cadr (car ne)))))
      (cond
        ((> bw 1e-9) (/ nw bw))
        ((> bh 1e-9) (/ nh bh))
        (T 1.0)
      )
    )
    (progn
      (princ (strcat "\n  [warn] bbox failed for " num-name " - using ratio 1.0"))
      1.0
    )
  )
)


; Main command.
(defun c:FixDupBlocks (/ all-names rec blkname basename ratio ss i ent ed sx sy sz total)
  (vl-load-com)
  (princ "\nFixDupBlocks - MicroStation duplicate-block fixer")

  ; Pass 1: collect all block names (uppercase) into a list.
  (setq all-names nil
        rec (tblnext "BLOCK" T))
  (while rec
    (setq blkname (cdr (assoc 2 rec)))
    (if blkname (setq all-names (cons (strcase blkname) all-names)))
    (setq rec (tblnext "BLOCK"))
  )

  ; Pass 2: find numbered variants and remap their INSERTs.
  (setq total 0
        rec (tblnext "BLOCK" T))

  (while rec
    (setq blkname (cdr (assoc 2 rec)))

    (if (and blkname (/= (substr blkname 1 1) "*"))
      (progn
        (setq basename (fdb-base-name blkname))
        (if (and basename (member (strcase basename) all-names))
          (progn
            (princ (strcat "\n  " blkname " -> " basename))
            (setq ratio (fdb-ratio basename blkname))
            (princ (strcat "  (x" (rtos ratio 2 4) ")"))

            (setq ss (ssget "_X" (list '(0 . "INSERT") (cons 2 blkname))))
            (if ss
              (progn
                (setq i 0)
                (while (< i (sslength ss))
                  (setq ent (ssname ss i)
                        ed  (entget ent))
                  (setq sx (if (assoc 41 ed) (cdr (assoc 41 ed)) 1.0)
                        sy (if (assoc 42 ed) (cdr (assoc 42 ed)) 1.0)
                        sz (if (assoc 43 ed) (cdr (assoc 43 ed)) 1.0))
                  (setq ed (fdb-setdxf ed  2  basename)
                        ed (fdb-setdxf ed 41  (* sx ratio))
                        ed (fdb-setdxf ed 42  (* sy ratio))
                        ed (fdb-setdxf ed 43  (* sz ratio)))
                  (entmod ed)
                  (entupd ent)
                  (setq total (1+ total)
                        i     (1+ i))
                )
              )
            )
          )
        )
      )
    )

    (setq rec (tblnext "BLOCK"))
  )

  (princ (strcat "\n\nDone - " (itoa total) " INSERT reference(s) updated."))
  (princ "\nRun PURGE -> All to remove unused block definitions.\n")
  (princ)
)
