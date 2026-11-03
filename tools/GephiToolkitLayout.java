import java.io.File;
import java.io.PrintWriter;
import java.util.Locale;

import org.gephi.graph.api.Graph;
import org.gephi.graph.api.GraphController;
import org.gephi.graph.api.GraphModel;
import org.gephi.graph.api.Node;
import org.gephi.io.importer.api.Container;
import org.gephi.io.importer.api.ImportController;
import org.gephi.io.processor.plugin.DefaultProcessor;
import org.gephi.layout.plugin.forceAtlas2.ForceAtlas2;
import org.gephi.layout.plugin.forceAtlas2.ForceAtlas2Builder;
import org.gephi.project.api.ProjectController;
import org.gephi.project.api.Workspace;
import org.openide.util.Lookup;

/**
 * Runs Gephi's own ForceAtlas2 headlessly and writes the coordinates to a TSV.
 *
 * This is the "use Gephi as much as you do" path: the Toolkit is Gephi's engine
 * without the window — same algorithm, same parameters — so the layout comes out
 * of Gephi itself rather than from a reimplementation.
 *
 * Usage: java -cp gephi-toolkit-all.jar GephiToolkitLayout.java in.gexf out.tsv [iters] [linlog]
 */
public class GephiToolkitLayout {

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("usage: GephiToolkitLayout <in.gexf> <out.tsv> [iterations] [linlog]");
            System.exit(2);
        }
        File input = new File(args[0]);
        File output = new File(args[1]);
        int iterations = args.length > 2 ? Integer.parseInt(args[2]) : 500;
        boolean linLog = args.length > 3 && Boolean.parseBoolean(args[3]);
        boolean randomize = args.length > 4 && Boolean.parseBoolean(args[4]);

        ProjectController projectController = Lookup.getDefault().lookup(ProjectController.class);
        projectController.newProject();
        Workspace workspace = projectController.getCurrentWorkspace();

        ImportController importController = Lookup.getDefault().lookup(ImportController.class);
        Container container = importController.importFile(input);
        if (container == null) {
            throw new IllegalStateException("could not import " + input);
        }
        importController.process(container, new DefaultProcessor(), workspace);

        GraphModel graphModel = Lookup.getDefault().lookup(GraphController.class)
                .getGraphModel(workspace);
        Graph graph = graphModel.getGraph();
        System.out.println("Gephi loaded " + graph.getNodeCount() + " nodes, "
                + graph.getEdgeCount() + " edges");

        if (randomize) {
            // Gephi's FA2 derives its forces from the current positions. Files that
            // already carry coordinates therefore *continue* that layout; with all
            // nodes at the origin the forces are zero. Spread them out first so the
            // run is genuinely from scratch. Fixed seed => reproducible.
            java.util.Random rng = new java.util.Random(42L);
            for (Node node : graph.getNodes()) {
                node.setX((float) (rng.nextGaussian() * 100.0));
                node.setY((float) (rng.nextGaussian() * 100.0));
            }
            System.out.println("randomised initial positions (seed 42)");
        }

        ForceAtlas2Builder builder = new ForceAtlas2Builder();
        ForceAtlas2 layout = builder.buildLayout();
        layout.setGraphModel(graphModel);
        layout.resetPropertiesValues();
        layout.setLinLogMode(linLog);
        layout.setAdjustSizes(true);       // Gephi's "prevent overlap"
        layout.setStrongGravityMode(true);
        layout.setGravity(1.0);
        layout.setScalingRatio(50.0);
        layout.setJitterTolerance(0.1);
        layout.setBarnesHutOptimize(true);

        long started = System.currentTimeMillis();
        layout.initAlgo();
        int done = 0;
        while (done < iterations && layout.canAlgo()) {
            layout.goAlgo();
            done++;
        }
        layout.endAlgo();
        System.out.println("Gephi ForceAtlas2: " + done + " iterations in "
                + (System.currentTimeMillis() - started) + " ms");

        try (PrintWriter writer = new PrintWriter(output, "UTF-8")) {
            writer.println("id\tx\ty");
            for (Node node : graph.getNodes()) {
                writer.println(node.getId() + "\t"
                        + String.format(Locale.ROOT, "%.6f", node.x()) + "\t"
                        + String.format(Locale.ROOT, "%.6f", node.y()));
            }
        }
        System.out.println("wrote " + output + " (" + graph.getNodeCount() + " positions)");
    }
}
