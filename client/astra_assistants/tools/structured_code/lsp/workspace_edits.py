def apply_workspace_edit(workspace_edit):
    if workspace_edit.changes and len(workspace_edit.changes) > 0:
        # Note one edit can have multiple changes
        for change in workspace_edit.changes:
            return apply_edits_to_file(change.uri, change.text_edits)
    elif workspace_edit.document_changes:
        for doc_change in workspace_edit.document_changes:
            return apply_edits_to_file(doc_change.text_document.uri, doc_change.edits)
    else:
        return None


def apply_edits_to_file(uri, text_edits):
    file_path = uri_to_file_path(uri)

    with open(file_path, 'r') as file:
        lines = file.read().splitlines()

    for edit in text_edits:
        start_line = edit.range.start.line
        start_char = edit.range.start.character
        end_line = edit.range.end.line
        end_char = edit.range.end.character

        # Apply the edit
        lines[start_line] = (
                lines[start_line][:start_char] + edit.new_text + lines[end_line][end_char:]
        )

        # If the edit spans multiple lines, handle those cases as well
        if start_line != end_line:
            for line in range(start_line + 1, end_line + 1):
                lines[line] = ''

    with open(file_path, 'w') as file:
        for line in lines:
            file.write(line + '\n')
    return "\n".join(lines)


def uri_to_file_path(uri):
    # Assuming the URI is in the format 'file:///path/to/file'
    return uri.replace('file://', '')
