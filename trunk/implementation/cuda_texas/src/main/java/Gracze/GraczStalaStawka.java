package Gracze;

public class GraczStalaStawka extends Gracz {

	@Override
	public double play(int i, double j) {
		bilans -= 20.0d - j;  
		
		return 20.0d;
	}

}
