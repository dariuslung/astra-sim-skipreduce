import re
from collections import defaultdict

def parse_logs_to_markdown(log_data: str) -> str:
    # Dictionary to hold parsed metrics for each system ID
    parsed_data = defaultdict(dict)
    
    # Regex patterns to capture system metrics
    finished_pattern = re.compile(r'sys\[(\d+)\] finished, \d+ cycles, exposed communication (\d+) cycles')
    metric_pattern = re.compile(r'sys\[(\d+)\]\,\s*([^:]+):\s*(\d+)')
    
    # Mapping log labels to your exact header titles
    metric_map = {
        "Wall time": "Wall Time",
        "Comm time": "Comm Time",
        "GPU time": "GPU Time",
        "Total compute-communication overlap": "Comp-Comm Overlap"
    }
    
    # Process logs line by line
    for line in log_data.strip().split('\n'):
        # Check for the line containing exposed communication
        finish_match = finished_pattern.search(line)
        if finish_match:
            sys_id = finish_match.group(1)
            exposed_comm = finish_match.group(2)
            parsed_data[sys_id]['Exposed Comm'] = exposed_comm
            continue
            
        # Check for standard key-value metrics
        metric_match = metric_pattern.search(line)
        if metric_match:
            sys_id = metric_match.group(1)
            metric_name = metric_match.group(2).strip()
            metric_value = metric_match.group(3).strip()
            
            # Map log label to target header string
            mapped_name = metric_map.get(metric_name, metric_name)
            parsed_data[sys_id][mapped_name] = metric_value

    # Exactly match your requested header format
    headers = [
        "System", 
        "Wall Time", 
        "Comm Time", 
        "GPU Time", 
        "Comp-Comm Overlap", 
        "Exposed Comm"
    ]
    
    # Sort system IDs numerically
    sorted_sys_ids = sorted(parsed_data.keys(), key=lambda x: int(x))
    
    # Build the Markdown table
    markdown_rows = []
    markdown_rows.append("| " + " | ".join(headers) + " |")
    markdown_rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for sys_id in sorted_sys_ids:
        row = [f"sys[{sys_id}]"]
        metrics = parsed_data[sys_id]
        for column in headers[1:]:
            row.append(metrics.get(column, "N/A"))
        markdown_rows.append("| " + " | ".join(row) + " |")
        
    return "\n".join(markdown_rows)

# Example Usage:
if __name__ == "__main__":
    raw_logs = """
[2026-07-15 06:34:29.602] [workload] [info] sys[2] finished, 231373820 cycles, exposed communication 231373810 cycles.
[2026-07-15 06:34:29.602] [statistics] [info] sys[2]. Post statistics processing start.
[2026-07-15 06:34:29.602] [statistics] [info] sys[2]. Post statistics processing end.
[2026-07-15 06:34:29.602] [statistics] [info] sys[2], Wall time: 231373820
[2026-07-15 06:34:29.602] [statistics] [info] sys[2], Comm time: 231373820
[2026-07-15 06:34:29.602] [statistics] [info] sys[2], GPU time: 10
[2026-07-15 06:34:29.602] [statistics] [info] sys[2], Total compute-communication overlap: 10
[2026-07-15 06:34:30.101] [workload] [info] sys[0] finished, 232050124 cycles, exposed communication 232050114 cycles.
[2026-07-15 06:34:30.101] [statistics] [info] sys[0]. Post statistics processing start.
[2026-07-15 06:34:30.101] [statistics] [info] sys[0]. Post statistics processing end.
[2026-07-15 06:34:30.101] [statistics] [info] sys[0], Wall time: 232050124
[2026-07-15 06:34:30.101] [statistics] [info] sys[0], GPU time: 10
[2026-07-15 06:34:30.101] [statistics] [info] sys[0], Comm time: 232050122
[2026-07-15 06:34:30.101] [statistics] [info] sys[0], Total compute-communication overlap: 8
[2026-07-15 06:34:30.734] [workload] [info] sys[1] finished, 233411510 cycles, exposed communication 233411500 cycles.
[2026-07-15 06:34:30.734] [statistics] [info] sys[1]. Post statistics processing start.
[2026-07-15 06:34:30.734] [statistics] [info] sys[1]. Post statistics processing end.
[2026-07-15 06:34:30.734] [statistics] [info] sys[1], Wall time: 233411510
[2026-07-15 06:34:30.734] [statistics] [info] sys[1], Comm time: 233411510
[2026-07-15 06:34:30.734] [statistics] [info] sys[1], GPU time: 10
[2026-07-15 06:34:30.734] [statistics] [info] sys[1], Total compute-communication overlap: 10
[2026-07-15 06:34:30.815] [workload] [info] sys[3] finished, 233685826 cycles, exposed communication 233685816 cycles.
[2026-07-15 06:34:30.815] [statistics] [info] sys[3]. Post statistics processing start.
[2026-07-15 06:34:30.815] [statistics] [info] sys[3]. Post statistics processing end.
[2026-07-15 06:34:30.815] [statistics] [info] sys[3], Wall time: 233685826
[2026-07-15 06:34:30.815] [statistics] [info] sys[3], GPU time: 10
[2026-07-15 06:34:30.815] [statistics] [info] sys[3], Comm time: 233685822
[2026-07-15 06:34:30.815] [statistics] [info] sys[3], Total compute-communication overlap: 6

    """
    
    markdown_table = parse_logs_to_markdown(raw_logs)
    print(markdown_table)