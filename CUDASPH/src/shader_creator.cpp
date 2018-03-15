
#include "shader_creator.h"

namespace Jex
{
	GLuint ShaderCreator::CreateShaderProgram(const char * vertexShader, const char * fragmentShader)
	{
		Shader shaders[2] = 
		{
			{vertexShader,   GL_VERTEX_SHADER,   0},
			{fragmentShader, GL_FRAGMENT_SHADER, 0}
		};

		GLuint shaderPrgm = glCreateProgram();

		for (size_t i = 0; i < 2; ++i)
		{
			Shader & shader = shaders[i];

			std::string srcStr = ReadShaderSource(shader.fileName);
			char * source = new char[srcStr.length()+1];
			for (size_t j = 0; j < srcStr.length(); ++j)
			{
				source[j] = srcStr[j];
			}
			source[srcStr.length()] = 0;

			GLuint _shader = glCreateShader(shader.type);
			glShaderSource(_shader, 1, (const char**)&source, 0);
			glCompileShader(_shader);

			GLint compileStatus;
			glGetShaderiv(_shader, GL_COMPILE_STATUS, &compileStatus);
			if (!compileStatus)
			{
				GLint logSize;
				glGetShaderiv(_shader, GL_INFO_LOG_LENGTH, &logSize);

				char * logMsg = new char[logSize];
				glGetShaderInfoLog(_shader, logSize, 0, logMsg);

				std::cerr << shader.fileName << " failed to compile :" << std::endl;
				std::cerr << logMsg << std::endl;

				delete[] logMsg;
				//exit(EXIT_FAILURE);
			}

			delete[] source;

			glAttachShader(shaderPrgm, _shader);
		}

		glLinkProgram(shaderPrgm);

		GLint linkStatus;
		glGetProgramiv(shaderPrgm, GL_LINK_STATUS, &linkStatus);

		if (!linkStatus)
		{
			GLint logSize;
			glGetProgramiv(shaderPrgm, GL_INFO_LOG_LENGTH, &logSize);

			char * logMsg = new char[logSize];
			glGetProgramInfoLog(shaderPrgm, logSize, 0, logMsg);

			std::cerr << "Shader program failed to link : " << std::endl;
			std::cerr << logMsg << std::endl;

			delete[] logMsg;

			//exit(EXIT_FAILURE);
		}

		m_currShaderPrgm = shaderPrgm;

		return shaderPrgm;
	}

	GLuint ShaderCreator::UseShaderProgram(GLuint shaderPrgm)
	{
		if (shaderPrgm == -1)
		{
			glUseProgram(m_currShaderPrgm);
			shaderPrgm = m_currShaderPrgm;
		}
		else
		{
			glUseProgram(shaderPrgm);
		}
		return shaderPrgm;
	}

}