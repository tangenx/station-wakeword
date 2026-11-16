package kws

// TrainingError identifies a quality rejection and the original enrollment
// recordings that need attention. TakeIndices refer to zero-based en.Takes.
type TrainingError struct {
	Code        string
	TakeIndices []int
	Seed        uint64
}

func (e *TrainingError) Error() string {
	switch e.Code {
	case "background_confusion":
		return "Обучение не завершено: модель путает слово с фоном или обычной речью. Запишите несколько похожих фраз без слова и повторите обучение."
	case "word_not_recognized":
		return "Обучение не завершено: модель не распознала проверочные записи слова. Прослушайте отмеченные записи и запишите слово ещё несколько раз с разных мест комнаты."
	default:
		return "Обучение не завершено: проверка качества модели не пройдена."
	}
}

func trainingQualityError(c calibration, original, check []int, seed uint64) error {
	if !c.Feasible {
		return &TrainingError{Code: "background_confusion", Seed: seed}
	}
	if c.Detected == 0 {
		indices := make([]int, len(check))
		for i, index := range check {
			indices[i] = original[index]
		}
		return &TrainingError{Code: "word_not_recognized", TakeIndices: indices, Seed: seed}
	}
	return nil
}
