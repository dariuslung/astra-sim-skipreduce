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
[2026-06-12 08:26:48.997] [workload] [info] sys[1] finished, 1752800 cycles, exposed communication 1752778 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[1]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[1]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[1], Wall time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[1], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[1], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[1], Total compute-communication overlap: 22
[2026-06-12 08:26:48.997] [workload] [info] sys[2] finished, 1752800 cycles, exposed communication 1752778 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[2]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[2]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[2], Wall time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[2], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[2], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[2], Total compute-communication overlap: 22
[2026-06-12 08:26:48.997] [workload] [info] sys[3] finished, 1752800 cycles, exposed communication 1752778 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[3]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[3]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[3], Wall time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[3], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[3], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[3], Total compute-communication overlap: 22
[2026-06-12 08:26:48.997] [workload] [info] sys[4] finished, 1752800 cycles, exposed communication 1752778 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[4]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[4]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[4], Wall time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[4], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[4], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[4], Total compute-communication overlap: 22
[2026-06-12 08:26:48.997] [workload] [info] sys[5] finished, 1752800 cycles, exposed communication 1752778 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[5]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[5]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[5], Wall time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[5], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[5], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[5], Total compute-communication overlap: 22
[2026-06-12 08:26:48.997] [workload] [info] sys[6] finished, 1752800 cycles, exposed communication 1752778 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[6]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[6]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[6], Wall time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[6], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[6], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[6], Total compute-communication overlap: 22
[2026-06-12 08:26:48.997] [workload] [info] sys[0] finished, 1752802 cycles, exposed communication 1752780 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[0]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[0]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[0], Wall time: 1752802
[2026-06-12 08:26:48.997] [statistics] [info] sys[0], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[0], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[0], Total compute-communication overlap: 20
[2026-06-12 08:26:48.997] [workload] [info] sys[7] finished, 1752804 cycles, exposed communication 1752782 cycles.
[2026-06-12 08:26:48.997] [statistics] [info] sys[7]. Post statistics processing start.
[2026-06-12 08:26:48.997] [statistics] [info] sys[7]. Post statistics processing end.
[2026-06-12 08:26:48.997] [statistics] [info] sys[7], Wall time: 1752804
[2026-06-12 08:26:48.997] [statistics] [info] sys[7], GPU time: 22
[2026-06-12 08:26:48.997] [statistics] [info] sys[7], Comm time: 1752800
[2026-06-12 08:26:48.997] [statistics] [info] sys[7], Total compute-communication overlap: 18
    """
    
    markdown_table = parse_logs_to_markdown(raw_logs)
    print(markdown_table)