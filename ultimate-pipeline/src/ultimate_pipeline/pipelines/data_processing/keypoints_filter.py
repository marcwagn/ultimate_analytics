import json
import sys

def filter_whitelisted_keypoints(meta_file: dict, class_title: str, keypoints_whitelist: list[str]) -> None:
    """
    Given a Supervisely annotations meta.json contant, filter out keypoints from a graph with a given class_title, 
    leaving only keypoints which names are whitelisted.
    Args:
        meta_file (dict): Supervisely meta.json content as a dictionary
        class_title (str): The class name to work on
        keypoints_whitelist (list): Keypoint labels to keep
    Note: The modification is done in place on the input dictionary.
    """
    graph_definition = next(filter(lambda x: x["title"]==class_title, meta_file["classes"]))
    if graph_definition["shape"] != "graph":
        raise ValueError(f"class must represent a graph, but is instead {graph_definition['shape']}")

    nodes_filtered = {}
    for (edge_key, edge_value) in graph_definition["geometry_config"]["nodes"].items():
        if edge_value["label"] in keypoints_whitelist:
            nodes_filtered[edge_key] = edge_value

    edges_filtered = []
    for edge_value in graph_definition["geometry_config"]["edges"]:
        if edge_value["dst"] in nodes_filtered and edge_value["src"] in nodes_filtered:
            edges_filtered.append(edge_value)

    graph_definition["geometry_config"]["nodes"] = nodes_filtered
    graph_definition["geometry_config"]["edges"] = edges_filtered

# Command line tool to filter Supervisely meta.json
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} path_to_meta_json")
        sys.exit(-1)

    with open(sys.argv[1], "r") as f:
        meta_content = json.load(f)

    # Leave only the keypoints below in Supervisely meta.json
    filter_whitelisted_keypoints(meta_content, "Field", ["TLC", "TRC", "TLF", "TRF"])
    print(json.dumps(meta_content))
