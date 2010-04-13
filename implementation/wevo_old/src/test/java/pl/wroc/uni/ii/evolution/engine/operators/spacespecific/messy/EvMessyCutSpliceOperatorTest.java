package pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy;

import junit.framework.TestCase;
import java.util.List;
import java.util.ArrayList;
import java.util.Arrays;
import pl.wroc.uni.ii.evolution.engine.individuals.EvMessyBinaryVectorIndividual;
import pl.wroc.uni.ii.evolution.engine.operators.spacespecific.messy.EvMessyCutSpliceOperatorTest;

public class EvMessyCutSpliceOperatorTest extends TestCase {
  
  public void testCut() {
    ArrayList<Integer> genes1 = new ArrayList<Integer>(3);
    ArrayList<Boolean> alleles1 = new ArrayList<Boolean>(3);
    genes1.add(0);alleles1.add(true);
    genes1.add(1);alleles1.add(true);
    genes1.add(2);alleles1.add(false);

    ArrayList<Integer> genes2 = new ArrayList<Integer>(4);
    ArrayList<Boolean> alleles2 = new ArrayList<Boolean>(4);
    genes2.add(0);alleles2.add(true);
    genes2.add(1);alleles2.add(true);
    genes2.add(3);alleles2.add(false);
    genes2.add(3);alleles2.add(false);

    EvMessyBinaryVectorIndividual individual1 =
      new EvMessyBinaryVectorIndividual(5, genes1, alleles1);    
    EvMessyBinaryVectorIndividual individual2 =
      new EvMessyBinaryVectorIndividual(5, genes2, alleles2);    
    
    // Only cut
    EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual> cutsplice =
      new EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual>(1.0, 0.0);
    
    ArrayList<EvMessyBinaryVectorIndividual> parents =
      new ArrayList<EvMessyBinaryVectorIndividual>();
    parents.add(individual1);
    parents.add(individual2);
    
    List<EvMessyBinaryVectorIndividual> children =
      new ArrayList<EvMessyBinaryVectorIndividual>();
    cutsplice.combine(parents, children);
    
    //Combine result size should be 4
    assertEquals(children.size(),4);
  }
  
  public void testSplice() {
    ArrayList<Integer> genes1 = new ArrayList<Integer>(3);
    ArrayList<Boolean> alleles1 = new ArrayList<Boolean>(3);
    genes1.add(0);alleles1.add(true);
    genes1.add(1);alleles1.add(true);
    genes1.add(2);alleles1.add(false);

    ArrayList<Integer> genes2 = new ArrayList<Integer>(4);
    ArrayList<Boolean> alleles2 = new ArrayList<Boolean>(4);
    genes2.add(0);alleles2.add(true);
    genes2.add(1);alleles2.add(true);
    genes2.add(3);alleles2.add(false);
    genes2.add(3);alleles2.add(false);

    EvMessyBinaryVectorIndividual individual1 =
      new EvMessyBinaryVectorIndividual(5, genes1, alleles1);    
    EvMessyBinaryVectorIndividual individual2 =
      new EvMessyBinaryVectorIndividual(5, genes2, alleles2);    
    
    //Only splice
    EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual> cutsplice =
      new EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual>(0.0, 1.0);
    
    ArrayList<EvMessyBinaryVectorIndividual> parents =
      new ArrayList<EvMessyBinaryVectorIndividual>();
    parents.add(individual1);
    parents.add(individual2);
    
    List<EvMessyBinaryVectorIndividual> children =
      new ArrayList<EvMessyBinaryVectorIndividual>();
    cutsplice.combine(parents, children);
    
    //Combine result size should be 1
    assertEquals(children.size(), 1);
  }

  public void testCutSplice() {
    ArrayList<Integer> genes1 = new ArrayList<Integer>(5);
    ArrayList<Boolean> alleles1 = new ArrayList<Boolean>();
    genes1.add(0);alleles1.add(true);
    genes1.add(1);alleles1.add(true);
    genes1.add(2);alleles1.add(false);
    genes1.add(2);alleles1.add(true);
    genes1.add(2);alleles1.add(true);
    genes1.add(4);alleles1.add(false);
    genes1.add(4);alleles1.add(false);

    ArrayList<Integer> genes2 = new ArrayList<Integer>(5);
    ArrayList<Boolean> alleles2 = new ArrayList<Boolean>();
    genes2.add(0);alleles2.add(true);
    genes2.add(1);alleles2.add(false);
    genes2.add(3);alleles2.add(true);
    genes2.add(3);alleles2.add(false);
    genes2.add(4);alleles2.add(true);
    genes2.add(4);alleles2.add(false);
    genes2.add(4);alleles2.add(true);
    genes2.add(4);alleles2.add(false);

    EvMessyBinaryVectorIndividual individual1 =
      new EvMessyBinaryVectorIndividual(5, genes1, alleles1);    
    EvMessyBinaryVectorIndividual individual2 =
      new EvMessyBinaryVectorIndividual(5, genes2, alleles2);    
    
    EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual> cutsplice =
      new EvMessyCutSpliceOperator<EvMessyBinaryVectorIndividual>(1.0, 1.0);
    
    ArrayList<EvMessyBinaryVectorIndividual> parents =
      new ArrayList<EvMessyBinaryVectorIndividual>();
    parents.add(individual1);
    parents.add(individual2);
    
    List<EvMessyBinaryVectorIndividual> children =
      new ArrayList<EvMessyBinaryVectorIndividual>();
    cutsplice.combine(parents, children);
    
    //Combine result size should be 2
    assertEquals(children.size(),2);
    
    //All genes should be the same
    ArrayList<Integer> parents_genes = new ArrayList<Integer>();
    parents_genes.addAll(parents.get(0).getGenes());
    parents_genes.addAll(parents.get(1).getGenes());
    Integer[] parents_genes_array = new Integer[parents_genes.size()];
    parents_genes.toArray(parents_genes_array);
    Arrays.sort(parents_genes_array);
    
    ArrayList<Integer> children_genes = new ArrayList<Integer>();
    children_genes.addAll(children.get(0).getGenes());
    children_genes.addAll(children.get(1).getGenes());
    Integer[] children_genes_array = new Integer[children_genes.size()];
    parents_genes.toArray(children_genes_array);
    Arrays.sort(children_genes_array);
    
    assertEquals(parents_genes.size(), children_genes.size());
    for (int i = 0; i < parents_genes.size(); i++)
      assertEquals(parents_genes_array[i],children_genes_array[i]);
        
    //All alleles should be the same
    ArrayList<Boolean> parents_alleles = new ArrayList<Boolean>();
    parents_alleles.addAll(parents.get(0).getAlleles());
    parents_alleles.addAll(parents.get(1).getAlleles());
    Boolean[] parents_alleles_array = new Boolean[parents_alleles.size()];
    parents_alleles.toArray(parents_alleles_array);
    Arrays.sort(parents_alleles_array);
    
    ArrayList<Boolean> children_alleles = new ArrayList<Boolean>();
    children_alleles.addAll(children.get(0).getAlleles());
    children_alleles.addAll(children.get(1).getAlleles());
    Boolean[] children_alleles_array = new Boolean[children_alleles.size()];
    parents_alleles.toArray(children_alleles_array);
    Arrays.sort(children_alleles_array);
    
    assertEquals(parents_alleles.size(), children_alleles.size());
    for (int i = 0; i < parents_alleles.size(); i++)
      assertEquals(parents_alleles_array[i], children_alleles_array[i]);
  }
  
}
