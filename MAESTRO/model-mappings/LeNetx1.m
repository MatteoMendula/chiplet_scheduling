Network lenet {
Layer CONV1 { 
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K 1,C 1,R 1,S 1,Y 32,X 32 }
}

Layer CONV2 { 
Type: CONV
Stride { X: 1, Y: 1 }
Dimensions { K 6,C 1,R 1,S 2,Y 28,X 28 }
}

Layer CONV4 { 
Type: CONV
Dimensions { K 16,C 6,R 1,S 2,Y 5,X 5}

}

Layer FC120 {
Type: CONV
Stride { X: 1, Y: 1 }		
Dimensions { K: 1, C: 120, R: 7, S: 7, Y: 7, X: 7 }
}

}
