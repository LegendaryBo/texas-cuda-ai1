package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.knaryvector.binaryvector.ecga;

import java.util.ArrayList;

/**
 * A structure for ECGA algorithm. It contains a partition of chromosome indexes
 * into blocks.
 * 
 * @author Marcin Golebiowski (xormus@gmail.com)
 */
public class EvECGAStructure {

  /**
   * contains blocks.
   */
  private ArrayList<EvBlock> blocks;


  /**
   * default constructor.
   */
  public EvECGAStructure() {
    blocks = new ArrayList<EvBlock>();
  }


  /**
   * ASdd <code>block</code> into Structure.
   * 
   * @param block to be added
   */
  public void addBlock(final EvBlock block) {
    blocks.add(block);
  }


  /**
   * Remove from structure block <code> first </code> and <code> second </code>.
   * 
   * @param first block to be removed
   * @param second block to be removed
   */
  public void remove(final EvBlock first, final EvBlock second) {
    for (int i = blocks.size() - 1; i >= 0; i--) {
      if (blocks.get(i) == first || blocks.get(i) == second) {
        blocks.remove(i);
      }
    }
  }


  /**
   * Gets set size.
   * 
   * @return set size
   */
  public int getSetCount() {
    return blocks.size();
  }


  /**
   * Gets structure rating.
   * 
   * @return rating
   */
  public double getRating() {
    double sum = 0.0;

    for (EvBlock b : blocks) {
      sum += b.getRating();
    }
    return sum;
  }


  /**
   * Gets the i'th block.
   * 
   * @param i index of block to get
   * @return i'th block
   */
  public EvBlock getBlock(final int i) {
    return blocks.get(i);
  }


  /**
   * {@inheritDoc}
   */
  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder();

    for (EvBlock block : blocks) {
      sb.append(block.toString());
      sb.append(" ");
    }
    return sb.toString();
  }


  /**
   * Convert structure into table of edges of size Z x 2.
   * Each block is represented by clique containing vertexes belonging to 
   * the block.
   * 
   * @return table of edges representing current structure
   */
  public int[][] getEdges() {
     
    // measuring table size
    int edge_count = 0;
    for (EvBlock block : blocks) {
      for (int i = 0; i < block.getSize() - 1; i++) {
        for (int k = i + 1; k < block.getSize(); k++) {
          edge_count++;
        }
      }
    }
    
    int[][] edges = new int[edge_count][2];
    
    int iter = 0;
    for (EvBlock block : blocks) {
      for (int i = 0; i < block.getSize() - 1; i++) {
        for (int k = i + 1; k < block.getSize(); k++) {
          edges[iter][0] = block.getElements().get(k);
          edges[iter][1] = block.getElements().get(i);
          iter++;
        }
      }
    }    
    
    return edges;
  }

  
  
}
