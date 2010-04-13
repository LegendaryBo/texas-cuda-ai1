package pl.wroc.uni.ii.evolution.grammar;

public class EvBNFParsingException extends Exception {

  /**
   * 
   */
  private static final long serialVersionUID = -7639142768119118963L;


  EvBNFParsingException() {
    super("BNFException occured.");
  }


  EvBNFParsingException(String message) {
    super("EvBNFException: " + message);
  }

}
